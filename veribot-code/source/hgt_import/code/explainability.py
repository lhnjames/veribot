"""
explainability.py  (v6 — VeriGraph 规范决策 DAG 与感应过滤版)
============================================================

更新：
  1. 引入了 LLM（Gemini API）证据充分性验证门控：
     - 历史代码调用外部 LLM 对关联内容作语义标注；这不是独立标签验证。
     - 分类为: sufficient_bot / supporting_but_insufficient / association_only / normal_human。
     - 拦截 normal_human (公共热门话题) 与 association_only (普通政治观点共现)，仅通过具备真正自动化/协同水军倾向的证据。
  2. 实现本地 JSON 缓存机制以防重复 API 耗时。
  3. 罕见度计算范围收紧为纯归纳式（Inductive）：
     `train_frequency` 仅统计与训练集节点（u < 8278）的连接数，排除测试集和验证集的信息影响，防止任何 transductive 边界泄露。
  4. 实现了 VeriGraph 决策 DAG 自动构建与生成接口 `build_verigraph_decision_dag()`：
     - 将每次正类判定导出为无环 DAG（含 Data Nodes, Computational Nodes, Atomic Claims, Derived Claims, Terminal Claim）。
     - 包含 LLM 语义充分性评估步骤节点。
"""
import os
import re
import json
import requests
import torch
from torch import Tensor
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

RT_PREFIX_RE = re.compile(r'^RT @\w+:\s*')
ROOT = os.environ.get("PROJECT_ROOT", "/path/to/project_root")

# ── 证据级别与筛选阈值定义 ───────────────────────────────────────────────────
TIER_A = "tier_a"   # hashtag / mention / 原创tweet
TIER_B = "tier_b"   # url
TIER_RT = "tier_rt" # RT 文本（最弱）

# 各级别的基本筛选门槛
TIER_THRESHOLDS = {
    TIER_A:  {"min_bot_count": 3, "min_wilson_lower": 0.55, "max_train_freq": 50},
    TIER_B:  {"min_bot_count": 5, "min_wilson_lower": 0.65, "max_train_freq": 100},
    TIER_RT: {"min_bot_count": 5, "min_wilson_lower": 0.75, "max_train_freq": 150},
}

# 仅供历史 TwiBot 正类边删除诊断：正向概率变化的筛选门槛。
# 单次删边可导致分布外输入，不能据此称为因果忠实或独立证据。
EDGE_SENSITIVITY_THRESHOLD = 0.03


def compute_wilson_lower_bound(successes: int, total: int, confidence: float = 0.90) -> float:
    """计算二项分布比例的威尔逊置信区间下界，防止小样本偏差"""
    if total == 0:
        return 0.0
    z = 1.645 if confidence == 0.90 else 1.28
    p = successes / total
    denominator = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    spread = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5)
    return max(0.0, (center - spread) / denominator)


def get_tweet_tier(entity_str: str) -> str:
    if RT_PREFIX_RE.match(entity_str.strip()):
        return TIER_RT
    return TIER_A


def get_entity_tier(entity_type: str, entity_str: str) -> str:
    if entity_type == "tweet":
        return get_tweet_tier(entity_str)
    elif entity_type == "url":
        return TIER_B
    else:  # hashtag, mention
        return TIER_A


# ── 可验证证据记录 ─────────────────────────────────────────────────────────────

@dataclass
class VerifiedEvidenceNode:
    """一条具有罕见度、威尔逊区间与 LLM 充分性校验的关联证据"""
    entity_type: str
    entity_str: str
    entity_idx: int
    tier: str
    attention: float
    train_frequency: int    # 该实体在训练集图中的度数（Inductive，反映罕见度）

    bot_training_uids: List[int] = field(default_factory=list)
    human_training_uids: List[int] = field(default_factory=list)
    delta_p: float = 0.0

    # LLM 语义标注字段
    sufficiency_label: str = "unverified"
    evidence_type: str = "unknown"
    llm_reasoning: str = ""

    @property
    def bot_count(self) -> int:
        return len(self.bot_training_uids)

    @property
    def human_count(self) -> int:
        return len(self.human_training_uids)

    @property
    def support(self) -> int:
        return self.bot_count + self.human_count

    @property
    def bot_ratio(self) -> float:
        return self.bot_count / max(self.support, 1)

    @property
    def wilson_lower_bound(self) -> float:
        """威尔逊比例置信区间下界"""
        return compute_wilson_lower_bound(self.bot_count, self.support, confidence=0.90)

    @property
    def rarity(self) -> float:
        """罕见度得分，度数越小越罕见"""
        return 1.0 / max(self.train_frequency, 1)

    @property
    def is_verifiable(self) -> bool:
        """
        Strict content-level verifiability gate:
          1. Disallow profile metadata anomalies as verifiable evidence.
          2. Disallow URL matches — url_density/shared-link 信号太弱：
             (a) 热门新闻/病毒式链接会被大量真人和bot同时转发，共享一个URL
                 不代表协同行为；(b) t.co短链接是逐条推文独立生成的，就算
                 两个账号转发同一篇原始文章，短链接字符串大概率都不同，
                 "精确匹配"既筛不掉弱信号，也捞不到真正同源的转发——两头
                 都不成立，不是调高门槛能解决的，直接排除这个证据类型。
          3. Check Wilson bounds and frequency thresholds.
          4. Ensure LLM (DeepSeek) classifies content as sufficient_bot.
        """
        if self.entity_type in ("profile", "url"):
            return False

        limits = TIER_THRESHOLDS.get(self.tier)
        if limits is not None:
            if self.bot_count < limits["min_bot_count"]:
                return False
            if self.wilson_lower_bound < limits["min_wilson_lower"]:
                return False
            if self.train_frequency > limits["max_train_freq"]:
                return False
                
        return self.sufficiency_label == "sufficient_bot"

    @property
    def has_positive_edge_sensitivity(self) -> bool:
        """
        历史 TwiBot 的方向性删边敏感性诊断：删边后 bot 概率须下降。
        这不是充分性、分布内可靠性或现实因果性的证明。
        """
        return self.delta_p >= EDGE_SENSITIVITY_THRESHOLD

    @property
    def evidence_score(self) -> float:
        tier_weight = {TIER_A: 1.0, TIER_B: 0.7, TIER_RT: 0.5}
        return self.wilson_lower_bound * self.rarity * tier_weight.get(self.tier, 0.5)

    def citation(self, max_bots: int = 3) -> str:
        cited = self.bot_training_uids[:max_bots]
        rest = self.bot_count - len(cited)
        s = f"训练集关联Bot uid=[{', '.join(str(u) for u in cited)}]"
        if rest > 0:
            s += f"（另有{rest}个）"
        return s


# ── 冻结模型删边敏感性诊断 ────────────────────────────────────────────────────

def compute_counterfactual_delta(
    model,
    data,
    user_idx: int,
    evidence_node: "VerifiedEvidenceNode",
    original_prob: float,
    device,
    x_dict_device: Optional[Dict[str, Tensor]] = None,
    edge_index_dict_device: Optional[Dict[Tuple, Tensor]] = None,
) -> float:
    """
    同时 mask 正向边 (user→entity) AND 反向边 (entity→user) 重新前向传播。

    性能说明：若调用方已经把整图的 x_dict/edge_index_dict 常驻在 GPU 上
    （评测/反验证循环里，每个用户要测多个候选证据，图本身不变），应通过
    x_dict_device/edge_index_dict_device 传入这份已在 GPU 上的引用——
    本函数只重新构造并上传"确实变化的那 2 个关系"（fwd_key/rev_key）的
    mask 后张量，其余 7 个关系直接复用已在 GPU 上的张量引用，不重新
    `.to(device)` 整图。若不传（如训练期偶尔独立调用的场景），退化为
    从 CPU 版 data 重新构建并上传整图，行为与之前一致，只是更慢。
    """
    if evidence_node.entity_type not in FWD_REV_ENTITY_MAP:
        raise ValueError(f"Unsupported edge intervention type: {evidence_node.entity_type}")

    fwd_key, rev_key = FWD_REV_ENTITY_MAP[evidence_node.entity_type]
    ent_idx = evidence_node.entity_idx

    if edge_index_dict_device is not None:
        edge_index_dict_masked = dict(edge_index_dict_device)  # 浅拷贝，只替换 2 个 key
        for k in (fwd_key, rev_key):
            edge = edge_index_dict_device[k]
            if k == fwd_key:
                keep = ~((edge[0] == user_idx) & (edge[1] == ent_idx))
            else:
                keep = ~((edge[0] == ent_idx) & (edge[1] == user_idx))
            edge_index_dict_masked[k] = edge[:, keep]
        x_dict = x_dict_device
    else:
        edge_index_dict_masked = {}
        for k, v in data.edge_index_dict.items():
            edge = v.cpu()
            if k == fwd_key:
                keep = ~((edge[0] == user_idx) & (edge[1] == ent_idx))
                edge_index_dict_masked[k] = edge[:, keep].to(device)
            elif k == rev_key:
                keep = ~((edge[0] == ent_idx) & (edge[1] == user_idx))
                edge_index_dict_masked[k] = edge[:, keep].to(device)
            else:
                edge_index_dict_masked[k] = v.to(device)
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}

    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict_masked)
        p_without = torch.sigmoid(out["bot_logit"])[user_idx].item()
        return original_prob - p_without


FWD_REV_ENTITY_MAP = {
    "url":     (('user', 'posted_url',     'url'),
                ('url',     'rev_posted_url',     'user')),
    "hashtag": (('user', 'used_hashtag',   'hashtag'),
                ('hashtag', 'rev_used_hashtag',   'user')),
    "tweet":   (('user', 'posted_tweet',   'tweet'),
                ('tweet',   'rev_posted_tweet',   'user')),
    "mention": (('user', 'posted_mention', 'mention'),
                ('mention', 'rev_posted_mention', 'user')),
}

_ENTITY_KEY_MULT = 10_000_000  # 大于任意实体节点数，保证 user*MULT+entity 组合唯一


def _build_masked_edge_index_dict(data, user_entity_pairs: List[Tuple[int, str, int]], device):
    by_type: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for u, etype, eidx in user_entity_pairs:
        if etype in FWD_REV_ENTITY_MAP:
            by_type[etype].append((u, eidx))

    edge_index_dict_masked = {k: v.to(device) for k, v in data.edge_index_dict.items()}

    for etype, pairs in by_type.items():
        if not pairs:
            continue
        fwd_key, rev_key = FWD_REV_ENTITY_MAP[etype]
        users_t = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        ents_t = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        key_set = users_t * _ENTITY_KEY_MULT + ents_t

        fwd_edge = edge_index_dict_masked[fwd_key]
        fwd_keys = fwd_edge[0].long() * _ENTITY_KEY_MULT + fwd_edge[1].long()
        keep_fwd = ~torch.isin(fwd_keys, key_set)
        edge_index_dict_masked[fwd_key] = fwd_edge[:, keep_fwd]

        rev_edge = edge_index_dict_masked[rev_key]
        rev_keys = rev_edge[1].long() * _ENTITY_KEY_MULT + rev_edge[0].long()
        keep_rev = ~torch.isin(rev_keys, key_set)
        edge_index_dict_masked[rev_key] = rev_edge[:, keep_rev]

    return edge_index_dict_masked


def compute_batched_counterfactual_delta(
    model,
    data,
    user_entity_pairs: List[Tuple[int, str, int]],
    device,
    orig_probs_all: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可微分、批量化的删边 Δp 计算：用于历史训练期一致性正则（区别于
    compute_counterfactual_delta 的推理期 no_grad 诊断用途）。

    一次性把 user_entity_pairs 中涉及的所有 (user, entity) 关联边同时从图中
    删除，只做一次「删边后前向传播」（而不是逐用户/逐证据单独跑一次前向传播），
    使 Δp 保持在计算图内、梯度可回传至模型参数。

    Args:
        user_entity_pairs: List of (user_idx, entity_type, entity_idx)
        orig_probs_all: 可选，预先算好的全图原始概率（detached，不含计算图）。
            训练循环里应传入本 step 主前向传播已经算出的概率，避免重复跑一次
            完整前向传播、同时持有两份全图激活值导致显存翻倍。若为 None，则
            退化为独立计算「原始」与「删边后」两次前向传播（用于训练循环之外
            的独立评测/消融场景）。
    Returns:
        (delta_p, user_idx_tensor)：delta_p[i] = 原始概率 - 删边后概率，
        对应 user_idx_tensor[i]。
    """
    edge_index_dict_masked = _build_masked_edge_index_dict(data, user_entity_pairs, device)
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    user_idx_tensor = torch.tensor([p[0] for p in user_entity_pairs], device=device, dtype=torch.long)

    if orig_probs_all is not None:
        p_orig = orig_probs_all[user_idx_tensor]
    else:
        edge_index_dict_orig = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        out_orig = model(x_dict, edge_index_dict_orig)
        p_orig = torch.sigmoid(out_orig["bot_logit"])[user_idx_tensor]

    out_masked = model(x_dict, edge_index_dict_masked)
    p_masked = torch.sigmoid(out_masked["bot_logit"])[user_idx_tensor]
    delta_p = p_orig - p_masked
    return delta_p, user_idx_tensor


def select_causally_verified_evidence(
    model,
    data,
    user_idx: int,
    candidates: List["VerifiedEvidenceNode"],
    original_prob: float,
    device,
    max_candidates_to_test: int = 5,
    x_dict_device: Optional[Dict[str, Tensor]] = None,
    edge_index_dict_device: Optional[Dict[Tuple, Tensor]] = None,
    require_magnitude: bool = True,
) -> Tuple[Optional["VerifiedEvidenceNode"], List["VerifiedEvidenceNode"], List["VerifiedEvidenceNode"]]:
    """
    历史函数名沿用；这里只检查候选内容匹配边的模型删边敏感性。
    候选先通过 Wilson/tier/LLM 规则并按 evidence_score 排序，再逐个用
    compute_counterfactual_delta 测量删边前后的 bot 概率差，将 delta_p
    写回每个候选节点。

    多条通过检查的候选可能高度相关，不能称为多条独立证据线。
    因此本函数返回**全部**通过指定方向性删边检查的候选（按 delta_p 降序），不只是
    最强的一条；调用方可以将全部结果呈现在 DAG 中，但不能称其相互独立。

    require_magnitude 控制筛选严格度：
      - True（默认）：必须满足 delta_p >= EDGE_SENSITIVITY_THRESHOLD，
        即删边后 bot 概率下降至少指定幅度。
      - False：只要求"方向与模型判断一致"，即 delta_p > 0（删掉这条 bot
        证据后，模型预测的 bot 概率确实下降，哪怕降幅很小），不再要求
        跨过固定的幅度阈值。它只排除删边后 bot 概率不下降的候选；
        不证明该边在其他输入分布或联合干预下仍重要。

    两种模式都允许空结果：如果一条候选都不满足所选标准，best 返回 None，
    verified_all 返回空列表，不强行展示一个和模型判断方向不一致的候选。

    Returns:
        (best, verified_all, tested)：best=正向 delta_p 最大的通过候选（可能为
        None）；verified_all=全部通过方向性删边检查的候选，按 delta_p 降序；
        tested=全部被测试过的候选（含未通过的，供报告展示"测过但没通过"）。

    候选池已经很小（内容匹配门槛筛过），所以逐个真实前向传播测试的开销可控，
    不需要像训练期 compute_batched_counterfactual_delta 那样批量化/可微分。
    """
    tested = candidates[:max_candidates_to_test]
    for node in tested:
        node.delta_p = compute_counterfactual_delta(
            model, data, user_idx, node, original_prob, device,
            x_dict_device=x_dict_device, edge_index_dict_device=edge_index_dict_device,
        )

    if require_magnitude:
        consistent = [n for n in tested if n.has_positive_edge_sensitivity]
    else:
        consistent = [n for n in tested if n.delta_p > 0]

    consistent.sort(key=lambda n: -n.delta_p)

    if not consistent:
        return None, [], tested

    return consistent[0], consistent, tested


# ── LLM 语义充分性 API 调用 ──────────────────────────────────────────────────

def query_deepseek_for_sufficiency(entity_type: str, entity_str: str, api_key: str = "") -> Dict[str, Any]:
    """使用 DeepSeek 官方 API 对内容实体进行垃圾信息、冒充、协同模板判别"""
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
You are an expert social media analyst and bot campaign investigator. Evaluate if the following entity (hashtag, mention, url, or tweet text), when shared/used identically by multiple accounts, represents a direct evidence of automated bot activity, coordinated spam, or trolling campaigns.

Entity Type: {entity_type}
Entity String: "{entity_str}"

You must classify it into one of these 4 labels:
1. "sufficient_bot": Direct, strong evidence of automation/bot campaign (e.g. spam/ad links, automated follow train/follow back templates, duplicate political coordinates, bot commands, or coordinated disinformation templates).
2. "supporting_but_insufficient": Supports bot hypothesis but too weak/generic on its own (e.g. a partisan retweet, a biased news URL, or a highly political hashtag).
3. "association_only": Indicates political or community association/views, but real people use this hashtag/mention frequently and organically.
4. "normal_human": General popular celebrity mentions, standard retweets of massive accounts, or public hashtags (e.g. @realDonaldTrump, #covid19, normal conversation text).

Respond strictly in JSON format with no markdown block around it:
{{
  "sufficiency_label": "sufficient_bot" | "supporting_but_insufficient" | "association_only" | "normal_human",
  "evidence_type": "spam_link" | "coordinated_template" | "politics_mention" | "public_hashtag" | "normal_chat",
  "reasoning": "Under 30 words explanation why this entity falls under the label"
}}
"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_object"
        },
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            text = res_json['choices'][0]['message']['content'].strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        else:
            print(f"  [DeepSeek API Warning] HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  [DeepSeek API Warning] {e}")
    
    return {
        "sufficiency_label": "supporting_but_insufficient",
        "evidence_type": "unknown",
        "reasoning": "DeepSeek API query failed or timed out."
    }


# ── 主函数：图遍历可验证证据提取 ──────────────────────────────────────────────

def extract_verifiable_evidence(
    model,
    data,
    user_idx: int,
    idx_to_url: Dict[int, str],
    idx_to_hashtag: Dict[int, str],
    idx_to_tweet: Dict[int, str],
    idx_to_mention: Dict[int, str],
    train_labels: torch.Tensor,
    train_boundary: int = 8278,
    top_k: int = 5,
    api_key: str = "",
    cache_dict: Optional[Dict[str, Any]] = None,
) -> List[VerifiedEvidenceNode]:
    """
    图遍历提取候选关联证据并进行 LLM 语义校验门控
    """
    edge_index_dict = data.edge_index_dict

    edge_offsets = {}
    current_offset = 0
    for k, v in edge_index_dict.items():
        edge_offsets[k] = current_offset
        current_offset += v.shape[1]

    url_edge     = data['user', 'posted_url',     'url'].edge_index.cpu()
    hashtag_edge = data['user', 'used_hashtag',   'hashtag'].edge_index.cpu()
    tweet_edge   = data['user', 'posted_tweet',   'tweet'].edge_index.cpu()
    mention_edge = data['user', 'posted_mention', 'mention'].edge_index.cpu()

    evidence_list: List[VerifiedEvidenceNode] = []

    saved_alpha = None
    saved_index = None
    if hasattr(model, "convs"):
        last_conv = model.convs[-1]
        saved_alpha = last_conv.saved_alpha.mean(dim=-1).cpu() if last_conv.saved_alpha is not None else None
        saved_index = last_conv.saved_index.cpu() if last_conv.saved_index is not None else None

    if saved_alpha is not None and saved_index is not None:
        matching_indices = (saved_index == user_idx).nonzero(as_tuple=True)[0]
        if len(matching_indices) > 0:
            entity_hits: Dict[Tuple[str, int], float] = {}

            for edge_idx in matching_indices:
                edge_idx = edge_idx.item()
                attn_val = saved_alpha[edge_idx].item()

                matched_edge_type = None
                local_edge_idx = 0
                for et, offset in sorted(edge_offsets.items(), key=lambda x: x[1]):
                    num_edges = edge_index_dict[et].shape[1]
                    if offset <= edge_idx < offset + num_edges:
                        matched_edge_type = et
                        local_edge_idx = edge_idx - offset
                        break

                if matched_edge_type is None:
                    continue

                type_map = {
                    ('url',     'rev_posted_url',     'user'): ('url',     edge_index_dict[matched_edge_type][0][local_edge_idx].item()),
                    ('hashtag', 'rev_used_hashtag',   'user'): ('hashtag', edge_index_dict[matched_edge_type][0][local_edge_idx].item()),
                    ('tweet',   'rev_posted_tweet',   'user'): ('tweet',   edge_index_dict[matched_edge_type][0][local_edge_idx].item()),
                    ('mention', 'rev_posted_mention', 'user'): ('mention', edge_index_dict[matched_edge_type][0][local_edge_idx].item()),
                }
                if matched_edge_type not in type_map:
                    continue

                key = type_map[matched_edge_type]
                if key not in entity_hits or entity_hits[key] < attn_val:
                    entity_hits[key] = attn_val

            for (entity_type, entity_idx), attn_val in sorted(entity_hits.items(), key=lambda x: -x[1])[:top_k * 4]:
                if entity_type == 'url':
                    entity_str = idx_to_url.get(entity_idx, f"url_{entity_idx}")
                    fwd_edge = url_edge
                elif entity_type == 'hashtag':
                    entity_str = idx_to_hashtag.get(entity_idx, f"#{entity_idx}")
                    fwd_edge = hashtag_edge
                elif entity_type == 'tweet':
                    entity_str = idx_to_tweet.get(entity_idx, f"tweet_{entity_idx}")
                    fwd_edge = tweet_edge
                elif entity_type == 'mention':
                    entity_str = idx_to_mention.get(entity_idx, f"@{entity_idx}")
                    fwd_edge = mention_edge
                else:
                    continue

                tier = get_entity_tier(entity_type, entity_str)
                mask = fwd_edge[1] == entity_idx
                connected_users = fwd_edge[0][mask].tolist()
                train_freq = sum(1 for u in connected_users if u < train_boundary)

                bot_uids, human_uids = [], []
                for u in connected_users:
                    if u == user_idx:
                        continue
                    if u < train_boundary:
                        lbl = train_labels[u].item()
                        if lbl == 1:
                            bot_uids.append(u)
                        elif lbl == 0:
                            human_uids.append(u)

                node = VerifiedEvidenceNode(
                    entity_type=entity_type,
                    entity_str=entity_str,
                    entity_idx=entity_idx,
                    tier=tier,
                    attention=attn_val,
                    train_frequency=train_freq,
                    bot_training_uids=bot_uids,
                    human_training_uids=human_uids,
                )
                evidence_list.append(node)

    # ── 穷举回退扫描 (如果通过 attention 没有找到任何候选证据) ──
    if not evidence_list:
        fallback_candidates = []
        for etype, fwd_edge, imap in [
            ('url',     url_edge,     idx_to_url),
            ('hashtag', hashtag_edge, idx_to_hashtag),
            ('tweet',   tweet_edge,   idx_to_tweet),
            ('mention', mention_edge, idx_to_mention),
        ]:
            user_mask = fwd_edge[0] == user_idx
            entity_indices = fwd_edge[1][user_mask].tolist()
            for ent_idx in set(entity_indices):
                ent_str = imap.get(ent_idx, f"{etype}_{ent_idx}")
                tier = get_entity_tier(etype, ent_str)
                ent_mask = fwd_edge[1] == ent_idx
                neighbor_users = fwd_edge[0][ent_mask].tolist()
                train_freq = sum(1 for u in neighbor_users if u < train_boundary)

                bot_uids, human_uids = [], []
                for u in neighbor_users:
                    if u == user_idx:
                        continue
                    if u < train_boundary:
                        lbl = train_labels[u].item()
                        if lbl == 1:
                            bot_uids.append(u)
                        elif lbl == 0:
                            human_uids.append(u)

                node = VerifiedEvidenceNode(
                    entity_type=etype, entity_str=ent_str,
                    entity_idx=ent_idx, tier=tier,
                    attention=0.001, train_frequency=train_freq,
                    # 不再用占位符 uid=0 伪造"训练集Bot参考账号"：如果确实没有
                    # 真实的已标注 Bot 邻居，bot_count 必须如实为 0，否则会在
                    # Wilson 下界与 tier 门槛判定中制造虚假证据支持。
                    bot_training_uids=bot_uids,
                    human_training_uids=human_uids,
                )
                fallback_candidates.append(node)
        if fallback_candidates:
            fallback_candidates.sort(key=lambda n: -n.evidence_score)
            evidence_list = fallback_candidates[:top_k]

    # ── 终极 Profile 兜底 (如果该账号没有任何内容关联边) ──
    # 该分支代表"完全没有可追溯的内容关联证据"，因此必须诚实地标注为不可验证，
    # 不能伪造 bot_training_uids 或直接标注 sufficient_bot——entity_type="profile"
    # 本身已经在 is_verifiable 中被硬性拒绝，这里的标签必须与之保持一致，
    # 否则会在 DAG/日志中输出与实际判定矛盾的虚假"证据确认"文案。
    if not evidence_list:
        node = VerifiedEvidenceNode(
            entity_type="profile",
            entity_str="No traceable content association found (no shared hashtag/mention/url/tweet)",
            entity_idx=-1,
            tier=TIER_A,
            attention=0.0,
            train_frequency=0,
            bot_training_uids=[],
            human_training_uids=[]
        )
        node.sufficiency_label = "unverified"
        node.evidence_type = "no_content_evidence"
        node.llm_reasoning = "No content entity available to query; account is graph-isolated from other labeled users."
        evidence_list.append(node)

    # ── 充分性标注 (绕过 API 直接在缓存填充) ──
    for node in evidence_list[:10]:
        if node.entity_type in ("profile", "url"):
            # entity_type=="profile"/"url" 在 is_verifiable 中被硬性拒绝
            # （url：共享链接信号太弱，且t.co短链天然难以精确匹配到真正同源
            # 转发，见 is_verifiable 文档），查询 LLM 没有意义（也会污染
            # 缓存），直接跳过。
            continue

        cache_key = f"{node.entity_type}:{node.entity_str}"
        if cache_dict is not None and cache_key in cache_dict:
            res = cache_dict[cache_key]
        else:
            res = query_deepseek_for_sufficiency(node.entity_type, node.entity_str, api_key)
            if cache_dict is not None:
                cache_dict[cache_key] = res

        # 安全默认值：任何解析失败/字段缺失的情况都必须保守地判为"不充分"，
        # 绝不能静默默认为 sufficient_bot，否则 API 响应异常会被自动放行为
        # 已验证证据，破坏"必须真正通过验证才能判定"的原则。
        node.sufficiency_label = res.get("sufficiency_label", "supporting_but_insufficient")
        node.evidence_type = res.get("evidence_type", "unknown")
        node.llm_reasoning = res.get("reasoning", "LLM annotation missing/malformed; conservatively treated as insufficient.")

    evidence_list.sort(key=lambda n: (-int(n.is_verifiable), -n.evidence_score))
    return evidence_list[:top_k]


# ── VeriGraph 决策证据 DAG 构建 ───────────────────────────────────────────────

def build_verigraph_decision_dag(
    user_idx: int,
    user_uid: str,
    verified_nodes: List[VerifiedEvidenceNode],
    hgt_prob: float,
    arm_value: float,
    threshold: float,
) -> Dict[str, Any]:
    """
    自动导出符合 VeriGraph 规范的无环决策证据 DAG (Decision Evidence DAG)。

    verified_nodes：通过方向性删边敏感性检查的候选记录。多条候选可能
    高度相关；并行 DAG 路径不证明统计独立、充分性或真实因果性。
    """
    if not verified_nodes:
        raise ValueError("verified_nodes 不能为空——需至少一条通过方向性删边检查的候选")
    if any(node.delta_p <= 0 for node in verified_nodes):
        raise ValueError("DAG 候选必须在删边后降低 bot 概率")

    d1 = {"id": "D1", "type": "data", "label": f"待测用户原始记录 uid=u{user_uid} (idx={user_idx})"}
    c6 = {"id": "C6", "type": "claim", "label": f"HGT基础预测 Bot 概率 = {hgt_prob:.3f} >= {threshold:.3f} (模型排序输出正类)"}
    nodes = [d1, c6]
    edges = [{"src": "D1", "dst": "C6", "type": "grounding"}]

    terminal_gate_srcs = ["C6"]  # 终端 Claim 的所有前提来源，最后统一接到 C7

    # 每条通过方向性检查的候选各生成一条并行来源路径，编号 i=1..N
    for i, node in enumerate(verified_nodes, start=1):
        d2 = {"id": f"D2_{i}", "type": "data",
              "label": f"共享内容实体#{i}: type={node.entity_type}, content='{node.entity_str}' (idx={node.entity_idx})"}
        ref_uids = node.bot_training_uids[:5]
        d_ref = {"id": f"D3_{i}", "type": "data",
                 "label": f"训练集参考关联账号记录#{i}: uids={ref_uids} (标注全为Bot)"}

        k1 = {"id": f"K1_{i}", "type": "computational", "label": f"参考关联账号计数 support={node.support}, bot_count={node.bot_count}"}
        k2 = {"id": f"K2_{i}", "type": "computational", "label": f"计算占比 bot_ratio = {node.bot_ratio*100:.1f}%"}
        k3 = {"id": f"K3_{i}", "type": "computational", "label": f"威尔逊置信下界计算 wilson_lower = {node.wilson_lower_bound:.3f} (confidence=90%)"}
        k4 = {"id": f"K4_{i}", "type": "computational", "label": f"归纳式训练集频次 train_frequency = {node.train_frequency}"}
        k5 = {"id": f"K5_{i}", "type": "computational", "label": f"LLM 语义充分性分析: label={node.sufficiency_label}, type={node.evidence_type}"}
        k6 = {"id": f"K6_{i}", "type": "computational",
              "label": f"删边敏感性测试: 移除该 user↔{node.entity_type} 边后重新前向传播, "
                        f"原 bot 概率减新概率 Δp = {node.delta_p:+.4f}；单次删边不证明因果性"}

        c1 = {"id": f"C1_{i}", "type": "claim", "label": f"待测用户发布/使用了内容实体#{i} '{node.entity_str}'"}
        c2 = {"id": f"C2_{i}", "type": "claim", "label": f"该内容实体被 {node.support} 个训练集已标注账号共同使用"}
        c3 = {"id": f"C3_{i}", "type": "claim", "label": f"共同使用该实体的账号中，被标注为 Bot 的账号占比置信下界为 {node.wilson_lower_bound:.3f}"}
        c4 = {"id": f"C4_{i}", "type": "claim", "label": f"该实体在训练集中使用频次为 {node.train_frequency} (满足罕见度规则)"}
        c5 = {"id": f"C5_{i}", "type": "claim", "label": f"内容实体#{i} '{node.entity_str}' 满足 TIER-{node.tier[-2:].upper()} 规则校验"}
        c_llm = {"id": f"C_LLM_{i}", "type": "claim", "label": f"LLM 对内容实体#{i}的意见（未验证标签正确性）: {node.llm_reasoning}"}
        c_cf = {"id": f"C_CF_{i}", "type": "claim",
                "label": f"方向性模型响应#{i}: 删除候选边后 bot 概率下降 Δp={node.delta_p:+.4f}；"
                          "尚未检验充分性、匹配控制或分布内可靠性"}

        nodes.extend([d2, d_ref, k1, k2, k3, k4, k5, k6, c1, c2, c3, c4, c5, c_llm, c_cf])
        edges.extend([
            {"src": "D1", "dst": f"C1_{i}", "type": "grounding"},
            {"src": f"D2_{i}", "dst": f"C1_{i}", "type": "grounding"},
            {"src": f"D2_{i}", "dst": f"C2_{i}", "type": "grounding"},
            {"src": f"D3_{i}", "dst": f"C2_{i}", "type": "grounding"},

            {"src": f"C2_{i}", "dst": f"K1_{i}", "type": "computational_input"},
            {"src": f"K1_{i}", "dst": f"K2_{i}", "type": "computational_derive"},
            {"src": f"K2_{i}", "dst": f"K3_{i}", "type": "computational_derive"},
            {"src": f"D2_{i}", "dst": f"K4_{i}", "type": "computational_input"},
            {"src": f"D2_{i}", "dst": f"K5_{i}", "type": "llm_evaluation"},
            {"src": "D1",       "dst": f"K6_{i}", "type": "computational_input"},
            {"src": f"D2_{i}", "dst": f"K6_{i}", "type": "computational_input"},

            {"src": f"K3_{i}", "dst": f"C3_{i}", "type": "grounding_calc"},
            {"src": f"K4_{i}", "dst": f"C4_{i}", "type": "grounding_calc"},
            {"src": f"K5_{i}", "dst": f"C_LLM_{i}", "type": "grounding_calc"},
            {"src": f"K6_{i}", "dst": f"C_CF_{i}", "type": "grounding_calc"},

            {"src": f"C1_{i}", "dst": f"C5_{i}", "type": "derivational"},
            {"src": f"C3_{i}", "dst": f"C5_{i}", "type": "derivational"},
            {"src": f"C4_{i}", "dst": f"C5_{i}", "type": "derivational"},
        ])
        terminal_gate_srcs.extend([f"C5_{i}", f"C_LLM_{i}", f"C_CF_{i}"])

    n_evidence = len(verified_nodes)
    c7 = {"id": "C7", "type": "claim",
          "label": f"HGT 输出 Bot；附 {n_evidence} 条来源候选及方向性删边敏感性结果。"
                    "这些结果不证明候选独立、充分或现实因果性。"}
    nodes.append(c7)
    for src in terminal_gate_srcs:
        edges.append({"src": src, "dst": "C7", "type": "derivational_gate"})

    best_node = verified_nodes[0]  # 按正向 Δp 降序排列，供 metadata 摘要用
    metadata = {
        "user_idx": user_idx,
        "user_uid": user_uid,
        "n_positive_edge_sensitivity_items": n_evidence,
        "evidence_summary": [
            {"entity_type": n.entity_type, "entity_str": n.entity_str, "entity_idx": n.entity_idx,
             "delta_p": n.delta_p, "positive_direction": n.delta_p > 0,
             "magnitude_threshold_passed": n.has_positive_edge_sensitivity}
            for n in verified_nodes
        ],
        "entity_idx": best_node.entity_idx,
        "entity_type": best_node.entity_type,
        "entity_str": best_node.entity_str,
        "attention_weight": best_node.attention,
        "delta_p": best_node.delta_p,
        "positive_direction": best_node.delta_p > 0,
        "magnitude_threshold_passed": best_node.has_positive_edge_sensitivity,
        "arm_value": arm_value
    }

    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges
    }
