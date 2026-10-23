# Jev-like Decision Model Experiments

本仓库用于研究：以小型 LLM 为例，如何适配为更准确、概率更可靠的决策模型。暂不涉及RLCD后训练流程。

输入是业务状态（state）、问题和候选项；输出是每个候选项的概率，以及最终选择。先围绕固定数据集验证效果，再考虑向新问题、新候选项泛化。

---

**当前进度：B0、G1、G2、P0、P1均已完成评估与复核。各训练组完成三个seed的训练、dev选模和test；详见 [G1 报告](reports/02_supervised_g1.md)、[G2 报告](reports/02_supervised_g2.md)、[P0 报告](reports/02_supervised_p0.md) 与 [P1 报告](reports/02_supervised_p1.md)。P2／C1及阶段3～4尚未开展。**

B0 官方 test 结果：**Accuracy 46.30%，NLL 1.3324，ECE15 0.2017**，每 case p50 延迟 330.91 ms。完整分组结果、运行记录和口径说明见 [基线报告](reports/01_baseline.md)；运行命令见 [阶段 1 说明](experiments/01_baseline/README.md)。

G1 官方 test 的三 seed Accuracy 为 **76.63% ± 1.57 个百分点**（均值 ± 样本标准差），平均提升 **30.33 个百分点**；NLL 均值 **0.5766**，ECE15 均值 **0.0712**。当前未合并 LoRA 的 p50 延迟约为 B0 的 3.45 倍，配置与限制见 [G1 报告](reports/02_supervised_g1.md)。

G2 官方 test 的三 seed Accuracy 为 **70.25% ± 4.74 个百分点**，较 B0 提升 **23.95 个百分点**，较 G1 低 **6.38 个百分点**；NLL 均值 **0.6928**，ECE15 均值 **0.0704**。完整配对区间、全量训练成本与恢复记录见 [G2 报告](reports/02_supervised_g2.md)。

P0 冻结原始 Backbone，只训练 **262,144** 个 Pointer 参数。三 seed Accuracy 为 **63.80% ± 0.90 个百分点**，较 B0 提升 **17.50 个百分点**；NLL 均值 **0.8284**，ECE15 均值 **0.0509**。选模、成本和结论边界见 [P0 报告](reports/02_supervised_p0.md)。

P1 联合训练 Pointer Head 与 LoRA，三 seed Accuracy 为 **76.30% ± 0.58 个百分点**，较 P0 提升 **12.50 个百分点**。相对 G1 的差为 -0.33 pp，配对case 95%区间为 **-1.05～+0.42 pp**，本轮没有观察到明确的准确率优势。NLL均值 **0.5588**，ECE15均值 **0.0589**；三个seed均按dev NLL选中第2轮，详见 [P1 报告](reports/02_supervised_p1.md)。

## 1. 项目整体路线

项目按以下四个阶段推进。Q1、Q2 是阶段 2 的两个研究问题，不是与四个阶段并列的项目目标。
本轮使用先回答「如何训练、用什么 Head」，再在后续阶段研究软标签和温度校准带来的增益。

| 阶段 | 做什么 | 与本轮的关系 |
| --- | --- | --- |
| 1. Baseline | 原始 Qwen3.5-0.8B，不训练 | B0 已完成，提供后续对照 |
| **2. 专项训练** | **仅用硬标签，比较 Backbone 适配方式和 Head 结构** | **本轮重点：Q1、Q2** |
| 3. 增加蒸馏／软标签监督 | 在相同训练数据上加入教师软分布 | 后续阶段，本轮不训练 |
| 4. 温度校准 | 在独立校准集上拟合温度 | 后续阶段，本轮不拟合 |

## 2. 基础模型与数据

| 项目 | 内容 |
| --- | --- |
| 初始模型 | Qwen3.5-0.8B，所有主实验从同一份 checkpoint 初始化 |
| 本地模型路径 | `/data/data_cold/Workshop/wangyueyi/llm_modelscope/qwen3.5-0.8b` |
| 数据集 | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) |
| 工作流 | 客服、发票处理、安全事件、Agent 轨迹观测 |
| 问题类型 | Noul（真假判断）、Choice（多选一）、Score（有序等级） |

官方数据包含 1,200 个训练 case、400 个测试 case，每个 case 有同一 state 下的 5 道问题。候选项的文字说明也是模型输入的一部分。


## 3. 本轮实验：阶段 2「专项训练」

本轮包括 Q1、Q2 的主实验、统一评估和必要的补充检查。所有训练组只使用硬标签；概率指标直接使用模型原始输出，不进行温度校准。

### 3.1 研究问题

#### Q1：Backbone 微调：冻结 Backbone、LoRA、全参数微调，哪种更有效？

在固定决策结构下，冻结预训练 Backbone 是否足够？LoRA 与全量微调分别带来多少收益？

- 保留 LM Head：比较 Zero-shot、LoRA、Full FT。
- 使用 Pointer Head：比较 Head Only、Head + LoRA、Head + Full FT。


#### Q2：Head 架构：复用 LM Head 的分类能力，还是重新设计 Decision Head？

在相同监督数据和 Backbone 适配方式下，比较 LM Head、Pointer Head 与 Candidate Head 的：

- 准确率和概率质量。
- 推理延迟、显存和训练成本。
- 动态候选支持、候选顺序鲁棒性、未知候选泛化。

### 3.2 主实验矩阵：一个 Baseline + 六组专项训练

#### 三种输出方式

| 输出方式 | 简单理解 | 主要取舍 |
| --- | --- | --- |
| LM Head 标签读出 | 给候选项分配 A/B/C 等单 token 标签，读取对应 logits 并归一化 | 保留预训练读出能力，但可能受标签和位置偏好影响 |
| Pointer Head | 提取决策位置与各候选位置的表示，通过匹配分数选择候选 | 支持动态候选；顺序拼接可能引入位置影响 |
| Candidate Head | 分别编码 `state + question + candidate`，经无位置编码的集合 Transformer、残差及共享标量 Head 打分 | 候选路径隔离，Head 允许集合内交互；重复编码成本较高 |

以上方式都可以直接输出概率，无需逐 token 生成 JSON。C1 参考本地 AgentJev 的完整候选头：投影到 256 维 → 两层无位置编码的集合 Transformer → 投影回 Backbone 维度并残差相加 → 共享标量打分器。独立的是候选编码路径，不是要求候选之间完全没有交互。具体输入信息、训练设置与参考项目的差异见 [AgentJev 代码核对](reports/agentjev_reference.md)。

#### 实验矩阵

保留已约定的七组编号。其中 B0 来自阶段 1，其余六组属于本轮阶段 2；它们共同构成回答 Q1、Q2 的对照矩阵。

| 编号 | 输出方式 | 训练范围 | 主要对应问题 | 参考项目与借鉴内容 |
| --- | --- | --- | --- | --- |
| B0 | LM Head | 不训练；复用阶段 1 基线 | Q1 的参考起点 | 通用 LM-token 判别读出；本项目的原始模型对照 |
| G1 | LM Head | Backbone LoRA；原 LM Head 保持冻结 | Q1、Q2 | [JevAny](https://github.com/SimpleJev/JevAny)：候选标签 token 的 logits 读出 + LoRA 适配 |
| G2 | LM Head | 文本模型全量微调 | Q1：与 G1 比较 | 沿用 G1 的 LM-token 读出路线；本项目扩展为 Full FT 对照 |
| P0 | Pointer Head | 冻结 Backbone，只训练新 Head | Q1：冻结表示是否够用 | [Kev](https://github.com/jaredpalmer/kev)：决策位置与候选位置的表示匹配；本项目扩展为 Head Only 对照 |
| **P1** | **Pointer Head** | **新 Head + Backbone LoRA** | **Q1、Q2；推荐主方案** | [Kev](https://github.com/jaredpalmer/kev)：Pointer Head 与 Backbone LoRA 联合训练 |
| P2 | Pointer Head | 新 Head + 文本 Backbone 全量微调 | Q1：与 P1 比较 | [Kev](https://github.com/jaredpalmer/kev)：Pointer 读出思路；本项目设置 Full FT 对照 |
| C1 | Candidate Head | 新 Head + Backbone LoRA | Q2：与 G1、P1 比较 | [AgentJev](https://github.com/malevrigns/agent-jev)：独立候选编码 + 无位置编码的集合 Transformer + 残差共享打分 |

这里的 Full FT 指文本路径的全量训练，本轮不涉及视觉任务。所有可训练专用 Head 都接受监督训练。

参考项目列记录本实验借鉴的具体部分，不表示项目只支持该路线，也不表示完整复现。各组采用本仓库统一的数据划分与评估协议；输入、Head 和训练配方的差异另行记录。

#### 如何解读对照

| 对照 | 回答什么 |
| --- | --- |
| B0 → G1 → G2 | 固定 LM 读出时，监督适配带来多少收益 |
| P0 → P1 → P2 | 固定 Pointer 结构时，是否需要修改 Backbone |
| G1 vs P1 | 相同 LoRA 适配下，专用 Pointer 是否有价值 |
| P1 vs C1 | 联合候选编码与独立候选编码的效果、成本差异 |

G1 与 P1 尽量使用相同输入布局和信息。C1 改变了编码方式，应当作为整个结构的比较，不能把全部收益归因于最后一层 Head。

### 3.3 数据划分与公平比较

已固定按工作流分层，从官方 train 中划分：

| 划分 | Case 数 | 用途 |
| --- | --- | --- |
| train | 960 | 更新模型参数 |
| dev | 120 | 选超参数与 checkpoint |
| calibration | 120 | 为阶段 4 预留；本轮不使用 |
| 官方 test | 400 | 方案冻结后的最终评估 |

本轮仍使用 960 个训练 case，预留的 calibration 不并入 train 或 dev，方便后续阶段在相同训练样本上比较增益。

约束：

- 按完整 case 划分，同一 state 下的 5 道问题不能跨集合。
- 固定数据集 revision、划分清单与随机种子；不要将 `all` 配置与各工作流数据重复拼接。
- 仅使用 state、问题和候选说明作为输入，排除 gold、`factors`、`label_agreement` 等标签或构造信息。
- 主实验使用相同硬标签、输入信息和截断规则，记录过长样本的处理方式。
- 硬标签交叉熵统一在候选范围内计算；若另做完整词表上的生成式 SFT，单独命名和报告。
- gold 并列最大时，使用固定选项身份规则处理；候选重排后标签身份保持一致。
- 给 LoRA 和 Full FT 可比的调参预算，学习率允许分别调整，记录训练步数、样本量与计算成本。
- 重要结论使用 3 个随机种子，并按 case 做配对 bootstrap 置信区间。
- 使用 dev 做方案选择，避免反复查看 test 来选择模型或补充训练数据。

本地配置显示 Qwen3.5-0.8B 混合了线性注意力与全注意力层。第一轮每道问题使用独立的因果输入；共享前缀、树形 attention 等优化后续单独验证。LoRA 覆盖范围应适配该混合架构，并记录具体模块。

本地模型的输入／输出 embedding 权重绑定，移除 LM 输出路径不一定减少同等规模的参数存储，需要实测显存与延迟。

### 3.4 评估与结果记录

| 维度 | 指标 | 如何理解 |
| --- | --- | --- |
| 选对答案 | 硬标签 Accuracy | 与 gold argmax 是否一致 |
| 概率质量 | 硬标签 NLL | 给正确标签分配了多少概率 |
| 分布拟合 | Soft CE／KL、Brier | 与完整 gold 分布有多接近 |
| 校准 | ECE | 置信度是否与硬标签正确率匹配 |
| 等级预测 | Score 期望值 MAE | 与 gold 期望分数的差距 |
| 成本 | 每 case 的 p50／p95 延迟、峰值显存、训练耗时、可训练参数量 | 提升是否值得 |


本轮主实验结果表，尚未运行的组保留空项；概率指标均为未经温度校准的结果：

| 编号 | GPU | Accuracy | NLL | Soft CE | Brier | ECE | p50 延迟 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 5090 | 46.30% | 1.3324 | 1.5062 | 0.3520 | 0.2017 | 330.91 ms/case | 1.73 GiB |
| G1（三 seed） | 5090 | 76.63% ± 1.57 pp | 0.5766 | 1.3339 | 0.1607 | 0.0712 | 1,140.28 ms/case | 1.78 GiB |
| G2（三 seed） | 5090 | 70.25% ± 4.74 pp | 0.6928 | 1.2844 | 0.1788 | 0.0704 | 328.49 ms/case | 1.73 GiB |
| P0（三 seed） | 5090 | 63.80% ± 0.90 pp | 0.8284 | 1.2569 | 0.1688 | 0.0509 | 1,144.31 ms/case | 1.74 GiB |
| P1（三 seed） | 4090 | 76.30% ± 0.58 pp | 0.5588 | 1.2732 | 0.1509 | 0.0589 | 254.49 ms/case | 1.78 GiB |
| P2 | — | — | — | — | — | — | — | — |
| C1 | — | — | — | — | — | — | — | — |

表中均为官方 test 结果，Brier 指 Soft Brier；显存指评估时 PyTorch 峰值 allocated，包含模型；B0保留整轮评估峰值，其余为test峰值。G1／G2／P0／P1 的 Accuracy 为三 seed 均值 ± 样本标准差（pp 为百分点），其余指标为均值，未挑选 test 最佳 seed；p50 为各run p50的均值。B0 run ID 为 `eval_20261021_170132_311730_seed42`，G1 为 `train_20261021_215958_618296`，G2 为 `train_20261022_104739_026131`，P0 为 `train_20261022_232846_262432`，P1 为 `train_20261023_102132_592199`。完整定义、置信区间和成本见各组报告；P2／C1尚未启动。

当前 P0 只训练262,144个新Head参数，相对B0提高17.50 pp；P1再加入LoRA后提高12.50 pp，支持本配方下适配Backbone的价值。G1与P1的Accuracy差区间包含0，尚未证明Pointer有准确率优势，也没有验证候选顺序或未知语义泛化。P0的低ECE不代表更高准确率；B0→P0同时改变监督与Head，不是单变量比较。正式test包含完整Backbone推理，缓存Head耗时不能当作部署延迟。

复盘时记录四件事：原来的假设、实际结果、证据支持的解释、下一步。不要只保留最好的单次成绩。

### 3.5 阶段 2 的补充检查与可选扩展

优先检查与 Q2 直接相关的候选顺序、表述改写与动态候选。未知候选语义泛化需要额外设计测试集；在完成前，不声称已经验证该能力。其余扩展按主实验结果选择，不作为本轮全部必做项。

| 实验 | 做法 | 目的 |
| --- | --- | --- |
| 候选顺序鲁棒性 | 打乱 Choice 选项，再映射回选项身份比较预测和概率 | 检查位置依赖 |
| 问题与选项改写 | 改写问题、替换选项 ID，保持语义不变 | 检查表面格式依赖 |
| 动态／未知候选 | 单独设计候选数量变化与候选语义留出集，包含正确答案为未见候选的样本 | 检验动态候选与语义泛化 |
| 数据效率 | 在代表模型上使用 10%、25%、50%、100% 训练 case | 判断少量监督是否足够 |
| 共享 vs 专项 | 四工作流共享模型，对比每工作流独立 LoRA／Head | 判断共享收益或负迁移 |
| 跨工作流迁移 | 训练三个工作流，留出第四个 | 检验新工作流适应能力 |
| 固定任务分类 Head | 针对固定问题训练 Linear／MLP 分类器 | 判断固定 schema 下简单结构是否够用 |
| LM Head 初始化／残差 | 分别测试标签权重行初始化、LM 分数加可训练残差 | 检验预训练读出先验的价值 |
| Score 有序损失 | 普通分类损失与有序损失对照 | 判断是否改善等级误差 |

候选重排后，输出概率随候选一起重排叫作置换等变；映射回选项身份后，决策应稳定。独立候选编码配合对称集合运算可以从结构上保证这种性质；顺序拼接候选的 Pointer 则需要实测。

原始测试集不足以充分证明未知候选语义泛化。更换 ID／名称主要检查格式鲁棒性，不能替代语义留出测试。Score 等级有顺序含义，重排测试需要单独设计。


## 4. 后续阶段：本轮仅保留方向

### 阶段 3：增加蒸馏／软标签监督

从阶段 2 中按 dev 表现选择少数代表模型，在相同训练样本上比较硬标签、硬标签 + gold 软分布、纯软分布训练。混合损失为：`L = (1 - λ) × CE(硬标签, p) + λ × CE(gold分布, p)`，权重只在 dev 上选择。

### 阶段 4：温度校准

冻结模型，在预留 calibration 上拟合正温度 T：`p = softmax(logits / T)`。优先尝试全局温度，比较校准前后的 NLL、Brier 和 ECE；拟合目标与分组方式在该阶段开始前确定。正标量温度不改变 argmax，因此不会提高 top-1 Accuracy。

## 5. 目录结构

```text
Jev/
├── README.md                         # 总体路线、本轮范围与实验说明
├── experiments/                      # 四个阶段的实验入口和配置
│   ├── 01_baseline/
│   │   ├── README.md                 # B0 的运行方式与注意事项
│   │   ├── run.py
│   │   └── configs/
│   │       └── b0.yaml
│   ├── 02_supervised/
│   │   ├── README.md                 # Q1、Q2 的实验与对照关系
│   │   ├── run.py                    # G1
│   │   ├── run_g2.py
│   │   ├── run_p0.py
│   │   ├── run_p1.py
│   │   ├── finish_g2.py               # 已保存结果的离线复核
│   │   ├── finish_p0.py
│   │   ├── finish_p1.py
│   │   └── configs/
│   │       ├── g1.yaml
│   │       ├── g2.yaml
│   │       ├── p0.yaml
│   │       ├── p1.yaml
│   │       ├── p2.yaml
│   │       └── c1.yaml
│   ├── 03_distillation/
│   │   ├── README.md                 # 硬／软监督对照
│   │   ├── run.py
│   │   └── configs/
│   └── 04_calibration/
│       ├── README.md                 # 温度拟合与校准前后对照
│       ├── run.py
│       └── configs/
├── src/
│   └── jev/                          # 各阶段复用的实现
│       ├── data/                     # 数据读取、标签转换与划分
│       ├── models/                   # Backbone 与三种 Head
│       ├── training/                 # Head Only、LoRA、Full FT、软标签损失
│       ├── evaluation/               # 预测、指标、成本与鲁棒性评估
│       └── calibration/              # 温度拟合与应用
├── data/
│   ├── raw/                          # 下载的原始数据或本地缓存
│   ├── splits/<split_id>/            # 共享划分清单，记录 case ID
│   │   ├── manifest.json             # 数据 revision、划分规则与 seed
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── calibration.jsonl
│   │   └── test.jsonl
│   └── teacher_labels/               # 后续新增教师标注；原始 gold 无需复制
├── outputs/                          # 运行自动生成的原始产物
│   ├── 01_baseline/b0/<run_id>/
│   ├── 02_supervised/<experiment_id>/<run_id>/
│   ├── 03_distillation/<experiment_id>/<run_id>/
│   └── 04_calibration/<experiment_id>/<run_id>/
└── reports/                          # 人工整理的结果与复盘
    ├── 01_baseline.md
    ├── 02_supervised.md              # 分别回答 Q1、Q2
    ├── 03_distillation.md
    └── 04_calibration.md
```

### 目录如何分工

| 目录 | 保存什么 | 原则 |
| --- | --- | --- |
| `experiments/` | 每阶段入口、配置与运行说明 | 入口调用共享实现，不复制整套训练代码 |
| `src/jev/` | 数据、模型、训练和评估代码 | 四阶段使用统一数据接口和指标定义 |
| `data/` | 数据缓存、划分与额外教师标注 | 所有阶段引用同一份划分，不各自随机切分 |
| `outputs/` | checkpoint、预测、logits、日志和指标 | 一次运行一个目录，避免覆盖旧结果 |
| `reports/` | 汇总表、误差案例和结论 | 引用具体 run ID，区分原始结果与人工解释 |

### 一次运行保存什么

例如：`outputs/02_supervised/p1/20261021_230000_seed42/`。`run_id` 使用北京时间的启动时间和随机种子，重名时追加编号。

```text
<run_id>/
├── config.yaml                       # 本次运行实际使用的完整配置快照
├── metadata.json                     # 代码版本、模型版本、split_id、环境与来源
├── run.log                           # 运行日志
├── source/                           # 本次运行代码与说明快照
├── splits/                           # 本次数据划分与来源清单快照
├── <split>/                          # 如 dev、test；各集合产物独立存放
│   ├── predictions.jsonl             # case/question/option 身份、gold 与预测概率
│   ├── logits.npz                    # 原始候选 logits 与有效候选 mask
│   ├── metrics.json                  # 总体、分组指标与置信区间
│   └── timing.json                   # 每 case 延迟与输入长度
├── checkpoints/                      # 训练产物；B0 不需要
└── temperature.json                  # 仅阶段 4 生成
```

B0 的实际运行命令、输入协议及检查方式见 [阶段 1 说明](experiments/01_baseline/README.md)。阶段 2～4 复用这一留存约定，具体产物以各阶段说明为准。

运行关系约定：

- **Q1、Q2 不各建一套实验目录。** P1 同时回答两个问题，训练和结果只保存一份，由报告中的不同对照引用。
- **B0 只放在阶段 1。** 阶段 2 的配置／报告引用基线 run ID，不复制或重新命名为另一个实验。
- 阶段 3 记录硬标签对照的 run ID、初始权重来源和软标签版本。若从阶段 2 checkpoint 继续训练，另行标明，避免与从原模型开始的对照混淆。
- 阶段 4 记录待校准模型的 run ID、logits 来源和拟合所用 split；保存温度与校准后结果，不复制整份模型权重。
- 配置、划分清单和复盘报告纳入版本管理；原始数据缓存、额外教师标注和 `outputs/` 中的大文件默认不纳入 Git，记录其版本、路径和校验信息。
- 原空目录 `baseline/` 已移除，Baseline 统一放在 `experiments/01_baseline/`。

## 6. 参考项目

| 项目 | 主要思路 | 本仓库参考点 |
| --- | --- | --- |
| [AgentJev](https://github.com/malevrigns/agent-jev) | Qwen Backbone + Candidate Head，直接输出候选分布 | 候选路径隔离与集合打分；[本地代码核对](reports/agentjev_reference.md) |
| [JevAny](https://github.com/SimpleJev/JevAny) | Pointer 与 LM token 读出，结合 LoRA 训练 | 输出结构对照、动态候选格式 |
| [Kev](https://github.com/Pendia/kev-Jev-Alternative) | 包含 Qwen3.5-0.8B 的 LoRA + Pointer 路线 | 混合架构适配、硬标签训练与校准 |
| [Luce](https://github.com/scienthoon/luce) | 根据任务选择打分结构，并进行专项训练和校准 | 固定／动态候选方案与概率评估 |
| [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 使用 decoder LLM 的 yes/no logits 做相关性打分 | 保留 LM 读出也能直接做判别 |
| [温度校准论文](https://proceedings.mlr.press/v70/guo17a.html) | 用独立数据拟合温度 | 校准流程与指标解释 |

参考项目采用的模型、数据和评估定义可能不同，公开分数不直接作为本项目的预期结果。

