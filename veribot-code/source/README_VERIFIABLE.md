# VeriBot 当前代码定位

主算法契约见 [research/VeriBot_v3_证据瓶颈目标与实施蓝图_2026-11-01.md](../research/VeriBot_v3_证据瓶颈目标与实施蓝图_2026-11-01.md)。主分类路径必须经过可核验事实、证据值和受约束推理层；原始特征、账号 ID、隐藏向量不得绕过证据层直接进入类别预测。主图数据集为 TwiBot-20、MGTAB、BotSim-24；Fox8-23 是辅助诊断。

## 当前固定模型的推理入口

`verifiable_bot/compact_evidence_runtime.py` 的 `CompactEvidenceRuntime` 从已保存的参考分布、20 项以内的证据配置和单调样条系数加载固定模型。`predict(facts, fact_names, account_ids)` 只接收按已核验词表排序的来源事实，不接收类别标签，也不拟合参数；它返回每项证据对 logit 的贡献、选中的原子来源事实、概率和类别。Fox8 加载时需传入 `research/fox8_20_concept_groups.json`，并核对验证集选择锁中的哈希。输入事实到原始记录/发布张量的来源核验由各数据集适配器与独立证书负责，运行时不把事实矩阵本身当作原始来源证书。

`verifiable_bot/compact_source_population.py` 不打开 Bot/Human 标签文件，直接从已审计的发布特征、原生关系事实缓存与 MGTAB 发布张量组装推理输入；逐文件哈希和参考词表必须与冻结模型一致。`scripts/verify_compact_inference_runtime.py` 在禁止读取类别标签文件的检查下重新加载 BotSim 12 概念、Fox8 20 概念、MGTAB 20 直接证据的三个固定种子。九个模型共 5,400 个测试账号的概率、每项贡献和选中来源事实与保存的证据卡完全一致，最大差均为零；[核对结果](../research/three_dataset_compact_runtime_replay_2026-11-03.json)。运行时还检查全部证据遮蔽后只剩截距。逐账号的单项关键证据与最少删除改判集合见[决策依赖审计](../research/three_dataset_compact_decision_dependence_2026-11-03.json)。

`scripts/verify_compact_raw_source_predictions.py` 在固定模型上重新读取 BotSim 原始用户/活动及回复父对象、Fox8 原始资料/推文、MGTAB 发布特征/关系边，并把核对过的事实送入统一运行时。三个数据集九个切分的来源值、概率及类别均与保存预测一致；[原始来源回放](../research/three_dataset_compact_raw_source_replay_2026-11-03.json)。Fox8 原生互动缓存按源文件哈希和独立事件审计绑定；MGTAB 的内容来源仅到发布的平均推文张量。

## 可复用基础

- `verifiable_bot/evidence_v2.py`：来源复算、事实验证与来源级响应审计。它尚未实现 v3 的参考背景、语义外部有效性和证据纠错。
- `verifiable_bot/fox8_adapter.py`、`botsim_adapter.py`：原始记录到模型输入的映射及局部来源核对。扩展到 Evidence Schema 时必须保留原始对象 ID、计算参数和来源层级。
- `scripts/audit_v3_fields.py`：检查三套发布数据中可用的原始字段；结果见项目的字段审计 JSON。
- `scripts/compare_paired_accuracy.py`：同切分、同账户集的配对分类评测；ACC 是报告指标，不是唯一优化目标。
- `scripts/gate0_shallow_tree.py`：深度 4 的可读字段捷径审计，已在三套数据运行三个种子。
- `scripts/verify_v3_fact_sources.py`：在 Fox8/BotSim 全部账户上从原始发布物复算现有派生字段，保存逐字段差错清单与文件哈希；不能替代证据外部有效性审查。
- `verifiable_bot/v3_evidence_reasoner.py`、`scripts/train_v3_computable.py`：无标签参考分布、A/B 分割、带符号非负加性推理的基础实现。选定紧凑模型已接入全员来源证书和已知测量错误纠正；独立语义真值仍缺，见[证据判决闭环](../research/VeriBot_v3_证据判决闭环与分层性能_2026-11-03.md)。
- `verifiable_bot/v3_monotone_spline_reasoner.py`、`scripts/screen_v3_monotone_spline.py`：保持 A/B 方向隔离，在 B 的无标签证据值上固定分位点并拟合非负分段斜率；三主图数据集仅用验证集筛选。`scripts/evaluate_selected_twibot20_spline.py` 和 `verify_selected_twibot20_spline.py` 对选中的 TwiBot-20 候选做官方测试与产物核对；结果见[筛选与测试记录](../research/VeriBot_v3_单调分段验证集筛选_2026-11-01.md)。
- `scripts/stress_twibot20_source_correction.py`：对无标签哈希选出的官方测试账号注入一个固定资料字段的测量错误，直接回查 ZIP 真值后纠正；记录冻结模型的前后错误及同数量直接标注账号对照。它是来源纠错压力测试，见[协议](../research/VeriBot_v3_TwiBot20来源纠错压力测试协议_2026-11-01.md)和[结果](../research/VeriBot_v3_TwiBot20来源纠错压力测试结果_2026-11-01.md)。
- `verifiable_bot/semantic_annotation.py`、`scripts/build_semantic_annotation_pilot.py`：从 TwiBot-20/BotSim 原始文本和可解析的回复关系生成类别盲、预测盲的语义证据标注试包，来源映射单独保存。`verifiable_bot/annotation_audit.py`、`scripts/audit_semantic_annotation_pilot.py` 验证盲化和双人答案一致率；材料与边界见[盲标试包说明](../research/VeriBot_v3_独立语义证据盲标试包_2026-11-01.md)。
- `scripts/verify_v3_saved_predictions.py`：读取保存的参考分布和推理器，核对输入文件、切分与逐账号概率；三套数据的九个候选运行均已通过。
- `scripts/evaluate_v3_group_reliance.py`、`verifiable_bot/group_reliance.py`：在保存的逐账号 logit 贡献上移除证据组，量化固定模型的输出依赖并检查贡献重建；三主图数据集结果见[证据组依赖检查](../research/VeriBot_v3_三主图数据集证据组依赖_2026-11-01.md)。
- `scripts/audit_native_graphs.py`：核对 TwiBot-20 官方 follow 关系与 BotSim 评论 parent ID 回复关系；直接流读 TwiBot-20 ZIP 中的 support ID。
- `scripts/build_botsim_native_facts.py`、`verifiable_bot/botsim_native_facts.py`：由 BotSim 原始 post/comment ID 建立账号回复图，保存可逐条查询的边来源；`train_v3_computable.py --native-graph-facts` 将 5 个图事实接入受约束推理候选。当前配对首测见 [BotSim 原生图接入记录](../research/VeriBot_v3_BotSim原生回复图接入_2026-11-01.md)。
- `verifiable_bot/botsim_source_certificate.py`、`scripts/verify_botsim_source_fact.py`：对指定账号重读每条相关评论与父对象，复算唯一出/入/互惠邻居和评论数，并与模型使用的图事实缓存逐值核对；示例见[data/README.md](../data/README.md)。
- `verifiable_bot/twibot20_facts.py`、`scripts/build_twibot20_facts.py`：从官方 ZIP 和原生 follow 关系生成 22 个可计算事实与来源索引；support 只用于解析图端点。
- `verifiable_bot/twibot20_source_certificate.py`、`scripts/verify_twibot20_source_fact.py`：按账户及事实名重读官方 ZIP，复算资料/推文/采样邻居事实或扫描原生 follow 声明复算图度数，出具含精确查询、操作、来源哈希的核验证书；示例见[data/README.md](../data/README.md)。
- `verifiable_bot/mgtab_source_certificate.py`、`scripts/verify_mgtab_source_fact.py`：按账户索引重读 MGTAB 发布的资料张量或类型化关系边，核对入/出度的 `log1p` 事实，证书标明只能达到发布张量层级。
- `scripts/train_v3_twibot20.py`、`verify_v3_twibot20_predictions.py`：官方 train/dev/test 上的第 0 关浅树和第 1 关可计算证据候选，以及保存产物的逐账号概率复核。当前候选结果见 [TwiBot-20 首测](../research/VeriBot_v3_原生图数据集切换与TwiBot20首测_2026-11-01.md)，不是完整 v3。

## 对照而非主算法

- `verifiable_bot/v2_detector.py` 与 `scripts/train_v2_detector.py` 是从原始特征直接分类的无证据瓶颈对照。
- `scripts/export_hypbot_paired.py` 提供 HypBot 配对基线。
- HGT、BotRGCN、BotDGT 的已有或后续实现可用于黑盒分类和事后解释对照。它们不自动形成主算法的可核验证据路径。
- Qwen SFT/GRPO、LLM 服务及 `verifiable_bot/v4/` 是其他探索分支。自然语言生成最多展示已经核验的结构化输出，不负责判定证据真值或直接给出主算法类别。

## 仍需验收的主算法组件

1. 将当前数据集事实与证据卡统一为可移植的 Evidence Schema；来源复算器与各自完整证书已经具备。
2. 对参考分布的异常度、可靠度和缺失状态取得独立语义核验，并检验预定上下文分层。
3. 为需预测的语义证据取得类别盲的证据级标注，再训练和校准预测器。
4. 对紧凑 A/B 加性模型补充同结构原始事实 GAM 对照，并保留来源级核验。
5. 在自然错误和实测人工成本上检验纠错收益；当前已完成已知来源错配和同数量直接标注对照。

当前已完成三数据集紧凑数值证据瓶颈、独立来源到判断回放及已知来源错配纠正；三个固定切分的 ACC 均数值超过同账号 HypBot。独立语义证据真值、自然错误发生率、真实审核成本和 OOD 效益仍需单独验收。MGTAB 的 16 项内容投影只达到发布平均推文张量层级，不能称为已验证的行为语义。
