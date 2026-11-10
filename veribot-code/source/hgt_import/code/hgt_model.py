import math
from typing import Dict, Tuple, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax

class VerifiableHGTConv(HGTConv):
    """
    Subclass of HGTConv that intercepts and saves attention weights (alpha)
    for explanation extraction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store latest attention details for explanation
        self.saved_alpha: Optional[Tensor] = None
        self.saved_index: Optional[Tensor] = None

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        # Calculate attention coefficients exactly like in HGTConv
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Save alpha and index (detached to avoid memory leaks)
        self.saved_alpha = alpha.detach()
        self.saved_index = index.detach()
        
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class VerifiableHGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for verifiable bot detection.
    Projects user features (1546D) and entity features (1D) into hidden_channels,
    runs HGT convolutions, and outputs classification logits.
    """
    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 in_channels: Dict[str, int], hidden_channels: int = 128,
                 out_channels: int = 1, num_layers: int = 2, heads: int = 4,
                 dropout: float = 0.3,
                 zero_init_extra_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # ── 1. Linear projections for node features ───────────────────────────
        self.lin_dict = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_channels)
            for node_type, in_dim in in_channels.items()
        })

        # 新增的实体先验特征(Wilson下界/训练集频次/tier)刚拼接进输入时是随机初始化
        # 权重接收的新增维度，会在训练早期给本就只有 1 维常数占位特征的实体节点
        # 引入额外噪声，扰乱优化轨迹（实测：即便调低辅助损失权重，测试集 ACC
        # 仍稳定低于旧模型 2-3 个百分点）。把这些新增维度对应的投影权重置零，
        # 使模型在训练起点上与旧的"1维常数特征"架构等价，只有当梯度下降确实
        # 发现这些先验特征有用时才会主动激活它们，而不是从一开始就被动引入噪声。
        if zero_init_extra_dims:
            with torch.no_grad():
                for node_type, n_zero in zero_init_extra_dims.items():
                    if node_type in self.lin_dict and n_zero > 0:
                        self.lin_dict[node_type].weight[:, -n_zero:] = 0.0

        # ── 2. Verifiable HGT layers ──────────────────────────────────────────
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = VerifiableHGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads
            )
            self.convs.append(conv)

        # ── 3. Classification heads on User node representation ──────────────
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.subtype_head = nn.Linear(hidden_channels, 6) # 6 subtypes

        # ── 4. 证据一致性辅助头 ────────────────────────────────────────────────
        # 读取与 classifier 相同的 user_emb，用真实标签联合训练，让"融入了
        # 实体先验特征(Wilson下界/训练集频次/tier)的表示"直接绑定进共享编码器，
        # 而不是像旧版那样在推理期外挂一个独立的符号门控/LLM 判断器。
        self.evidence_head = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        # Project all input features to hidden_channels
        h_dict = {
            node_type: F.gelu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Convolve through HGT layers, with an EXPLICIT residual add of the previous layer's
        # representation at every layer (in addition to HGTConv's own internal learned skip
        # parameter). This guarantees a node with zero incoming edges of every relation at some
        # layer -- e.g. one of this graph's isolated (no-neighbor) accounts -- still carries its
        # own projected features through to the classifier head unchanged, rather than depending
        # entirely on a learned skip weight that could in principle be driven toward the
        # message-aggregation term during training. Isolated nodes must never lose representation.
        for conv in self.convs:
            h_prev = h_dict
            h_new = conv(h_dict, edge_index_dict)
            h_dict = {
                k: F.dropout(F.gelu(v + h_prev[k]), p=self.dropout, training=self.training)
                for k, v in h_new.items() if v is not None
            }

        user_emb = h_dict['user']
        logits = self.classifier(user_emb).squeeze(-1) # [N]
        subtype_logits = self.subtype_head(user_emb)   # [N, 6]
        evidence_logits = self.evidence_head(user_emb).squeeze(-1)  # [N]

        return {
            "bot_logit": logits,
            "subtype_logit": subtype_logits,
            "evidence_logit": evidence_logits,
            "user_emb": user_emb
        }
