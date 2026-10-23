# AgentJev 本地代码核对

参考代码：`/data/dataset2/Workshop/wangyueyi/Projects/agent-jev/typed_decisions/experiment.py`，同时阅读了 `prepare_data.py`、`protocol.json`、`agentjev/model.py`、`agentjev/encoders.py` 和 `jev_service/contract.py`。

结论：这套代码很适合参考 C1 的结构与训练组织，但不是原始 LLM 的 Zero-shot 实验。它从已有 phase4 checkpoint 继续训练，使用软标签损失，再进行温度校准，覆盖了本项目多个阶段的内容。

## 1. 可以借鉴什么

| 部分 | AgentJev 的实现 | 本项目如何使用 |
| --- | --- | --- |
| 数据划分 | 先按 case 切分，再展开问题；960/120/120/400 | 已采用相同数量与隔离原则，继续固定本项目的划分清单 |
| 候选编码 | 每个 `state + question + candidate` 是独立因果路径，读取候选末 token 表示 | 作为 C1 的 Backbone 编码方式参考 |
| Candidate Head | 投影到 256 维 → 无位置编码的两层集合 Transformer → 投影回原维度并残差相加 → 共享标量打分器 | C1 拟采用这种完整结构，而非只用一个共享 MLP |
| 动态候选 | `cand_mask` 标识有效候选，分布只在有效候选上归一化 | 训练、推理和指标中统一候选 mask 与身份映射 |
| 优化组织 | Backbone 与 Head 不同学习率、梯度累积、梯度裁剪、warmup + cosine | 阶段 2 的实现参考，具体超参数由本项目 dev 选择 |
| 选模 | 包含 step 0，按 dev 指标选择 checkpoint，之后才做 test | 借鉴先选模再最终评估的流程 |
| 结果留存 | 保存原始 logits、校准前后预测和选模记录 | 已有 B0 保存 logits；后续阶段沿用 |

**需要修正此前对 C1 的简化描述：独立的是每个候选的 Backbone 路径，Head 仍允许候选集合内部交互。** 集合 Transformer 不加位置编码，配合共享打分器，在推理模式下具有置换等变结构；仍需验证完整输入、mask 和数值精度下的重排结果。

## 2. 不能直接复制的实验条件

| 条件 | 本地 AgentJev 实验 | 本项目 |
| --- | --- | --- |
| Backbone | Qwen3-0.6B-Base | 指定的 Qwen3.5-0.8B 本地 checkpoint |
| 初始化 | 再加载 `runs/phase4/final.pt`，已有任务适配 | 七组主实验从同一原始 checkpoint 初始化 |
| 训练范围 | 当前脚本对模型做全参数优化 | 对照 Head Only、LoRA、Full FT |
| 监督 | Soft CE + 0.1 × Soft Brier | 阶段 2 仅硬标签 CE；软监督留到阶段 3 |
| 选模指标 | dev Soft CE | 阶段 2 实现前固定选模标准，不能无说明地混用 |
| 温度校准 | 每种 primitive 一个正温度，在独立 calibration 上拟合 | 留到阶段 4，本轮不拟合 |
| 编码长度 | 每候选路径最多 2,048 token | 当前 B0 每问题最多 8,192 token；都拒绝超长而非静默截断 |
| 输入语义字段 | 使用候选 description，不输入候选 key；不使用 chat template | B0 使用 chat template 和 `letter + option_id + description` |

如果 C1 完全照搬参考输入格式，除了 Head，也改变了输入内容与格式。因此主对照要保持语义信息一致；如要测“只用 description”的参考格式，应单独记录为结构／输入消融。

参考 TreeEncoder 依赖全注意力及祖先 attention mask。Qwen3.5 有线性注意力层，不能直接照搬树形 mask 优化；先采用普通独立路径验证正确性。

软标签实验也不应同时无说明地加入 Brier 辅助损失。先在相同损失形式下比较硬／软监督，再把 Brier 作为单独消融，才能区分收益来源。

## 3. 数据是否一致

两边记录的 dataset revision 不同：

- 本项目：`d9eda7f31ba227333213a1787f2c78fdaa4090c8`。
- 参考项目：`ea9306458d6e9563628369a3d1e72e362fb381d2`。

核对参考项目下载元数据中的 LFS SHA256 与本项目的文件清单，**四工作流的八个 train/test Parquet 哈希全部一致**。官方 test 的 400 个 case ID 也完全一致。

但训练侧的划分不同：本项目 seed 42，参考项目按 `SHA256("20260921:" + case_id)` 排序。数量相同不表示样本清单相同。

| 划分 | 本项目数量 | 参考项目数量 | 共有 case 数 |
| --- | --- | --- | --- |
| train | 960 | 960 | 765 |
| dev | 120 | 120 | 13 |
| calibration | 120 | 120 | 6 |
| test | 400 | 400 | 400 |

本项目继续使用现有固定清单，不为贴近参考成绩而重新划分。

## 4. 用现有 B0 预测核对指标

只提取并执行参考脚本的纯 `metrics()` 函数，对已保存的 B0 概率离线计分；没有重新推理、训练或拟合温度，也没有修改原 B0 指标。

在保持 B0 原候选顺序时，参考函数算出的 Accuracy、Soft CE、Soft Brier 和 Score MAE 与本项目一致（浮点累加误差小于 `1e-12`）。额外得到 `gold_mass_at_selected_option`，可用来观察选中候选在 gold 软分布中获得了多少概率质量。

| B0 official test，同一份预测 | 本项目主口径 | 参考指标，保留 B0 顺序 | 参考指标及 Boolean 顺序 |
| --- | --- | --- | --- |
| Accuracy | **46.30%** | 46.30% | 46.80% |
| Soft CE | 1.5062 | 1.5062 | 1.5062 |
| Soft Brier（候选误差求和） | 0.3520 | 0.3520 | 0.3520 |
| ECE | 0.2017，15 bins | 0.1998，10 bins | 0.1948，10 bins |
| Score 期望值 MAE | 0.7047 | 0.7047 | 0.7047 |
| Gold mass at selected option | 主表未列 | 0.4059 | 0.4059 |

**46.80% 不是模型提升。** 参考项目把 Boolean 候选排为 `true, false`，再用 NumPy 首项 argmax 处理 gold 并列；本项目按稳定选项身份处理并列。这十道 Boolean gold 并列题的计分差异，使同一份预测多算对十题。主 B0 仍保留 **46.30%**。

这说明比较项目时必须对齐候选身份、并列规则和 ECE 分箱；不能只比较一个总分。上表也不是 AgentJev 模型成绩，而是对本项目 B0 预测的重新计分。

## 5. 后续落实到阶段 2

1. C1 参考“独立候选路径 + 无位置编码的集合 Transformer + 残差标量 Head”。不加载对方已有 phase4 checkpoint。
2. 与其余主实验保持相同 Backbone 起点、硬标签、数据清单与指标定义。
3. 借鉴分组学习率、候选 mask、梯度累积和 dev 选模；具体设置在实现前固定。
4. Soft CE、Brier 辅助损失、温度校准分别放在后续对应阶段或独立消融。

离线核对记录：[agentjev_reference_rescoring.json](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/agentjev_reference_rescoring.json)。其中保存了参考脚本哈希、数据哈希对照、划分交集和 dev/test 两套计分结果。
