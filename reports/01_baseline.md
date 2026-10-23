# 阶段 1：B0 基线结果

B0 已完成：原始 Qwen3.5-0.8B 不做专项训练，使用原 LM Head 的候选字母 logits 做决策。官方 test 上答对 **926 / 2,000 题，Accuracy 为 46.30%**。

本结果表示与数据集教师硬标签的一致性，不是人工评估的业务正确率，也不代表模型所有提示方式或生成式推理下的能力。

## 1. 本次运行

| 项目 | 记录 |
| --- | --- |
| Run ID | `eval_20261021_170132_311730_seed42` |
| 运行时间 | 2026-10-21 17:01:32～17:04:55，北京时间 |
| 初始模型 | `/data/data_cold/Workshop/wangyueyi/llm_modelscope/qwen3.5-0.8b` |
| 实际加载 | 文本部分，752,393,024 参数；不加载视觉模块 |
| 可训练参数 | 0；无训练、无生成 token、无温度拟合 |
| 数据版本 | `d9eda7f31ba227333213a1787f2c78fdaa4090c8` |
| 数据划分 | train 960 / dev 120 / calibration 120 / official test 400 cases |
| 划分目录 | `data/splits/typed_decisions_d9eda7f_seed42_v1/` |
| 正式评估 | dev 600 题、test 2,000 题；无截断、无跳过 |
| GPU | 物理 GPU 1，NVIDIA GeForce RTX 5090 D v2 |
| 精度与读出 | BF16，SDPA，每 case 五道独立问题组成一批，候选字母 logits 做 FP32 softmax |
| 环境 | Python 3.13.11；PyTorch 2.9.0+cu128；Transformers 5.14.1；NumPy 2.2.6；PyArrow 24.0.0 |

Conda 环境名是 `torch29-cu130`，实际 PyTorch 构建为 `2.9.0+cu128`，以运行 metadata 为准。未安装 FLA / causal-conv1d，使用 Transformers 的原生 PyTorch 路径。

## 2. 总体结果

以下均为未经温度校准的结果。概率指标越低越好。

| 指标 | dev | official test |
| --- | --- | --- |
| Cases / 决策数 | 120 / 600 | 400 / 2,000 |
| 正确数 | 245 | 926 |
| Accuracy | 40.83% | **46.30%** |
| Accuracy 95% 区间 | 35.83%～45.83% | 43.55%～49.05% |
| Hard NLL | 1.4703 | **1.3324** |
| Soft CE | 1.5622 | 1.5062 |
| KL(gold ‖ prediction) | 0.7985 | 0.7389 |
| Soft Brier | 0.3815 | 0.3520 |
| Hard Brier | 0.7834 | 0.7224 |
| ECE，15 等宽分箱 | 0.2518 | **0.2017** |
| Score 期望值 MAE | 0.7004 | 0.7047 |
| 每 case p50 延迟 | 330.12 ms | **330.91 ms** |
| 每 case p95 延迟 | 359.49 ms | 366.84 ms |

Accuracy 区间按完整 case 重采样，2,000 次 bootstrap，seed 42。同一 case 的五道题一起进入重采样，不视为五个独立样本。B0 本身不训练，仅运行一个固定 seed；后续训练组再比较多个 seed。

显存峰值：allocated **1.73 GiB**，reserved **2.10 GiB**。这是包含文本模型的 PyTorch allocator 记录，不是整张显卡的总占用。

延迟按一个 case 的全部五道题统计，包含分词、数据传输、同步推理和概率搬回 CPU，排除模型加载、预检、预热、指标计算与文件写入。它是当前原生实现的耗时，不能直接与不同硬件或其他 serving 实现比较。

## 3. Test 分组结果

### 按工作流

| 工作流 | 决策数 | Accuracy | Hard NLL | Soft Brier | ECE15 |
| --- | --- | --- | --- | --- | --- |
| Agent 轨迹观测 | 500 | 34.40% | 1.5518 | 0.3307 | 0.2182 |
| 客服 | 500 | 61.80% | 1.0811 | 0.2409 | 0.1361 |
| 发票处理 | 500 | 30.60% | 1.5717 | 0.5745 | 0.3501 |
| 安全事件 | 500 | 58.40% | 1.1249 | 0.2619 | 0.2272 |

### 按问题类型

| 类型 | 决策数 | Accuracy | Hard NLL | Soft Brier | ECE15 |
| --- | --- | --- | --- | --- | --- |
| Choice | 600 | 57.00% | 1.2287 | 0.2348 | 0.0829 |
| Noul | 600 | 49.33% | 1.1670 | 0.5185 | 0.3903 |
| Score | 800 | 36.00% | 1.5342 | 0.3150 | 0.2098 |

## 4. 如何解读这次结果

- 不同工作流的表现差异明显，后续阶段 2 应同时报告总体与分组指标。
- 在当前读出协议下，Score 比 Choice 弱；Noul 的硬标签 ECE 较高。不能据此直接推断冻结 Backbone 没有相关信息，仍需 P0／P1／P2 等对照。
- 平均最高候选概率为 66.27%，高于 46.30% 的硬标签 Accuracy。置信度至少 0.9 的 450 题中，有 158 题与 gold argmax 不同。这提示需要同时观察概率质量，不能只看是否选对。
- 举例：`agent_trace_observability_000000 / needs_review` 的预测为 `true`，概率 0.9241，gold argmax 为 `false`。这只是与教师标签不一致的案例，不代表经过人工确认的业务错误。
- 当前配置是冻结的 B0 对照。后续阶段只通过 train/dev 选择训练设置，不根据上述 test 误差逐条修改提示或补造训练样本。

## 5. 指标口径与数值限制

- 硬标签取 gold 的 argmax；gold 舍入误差在校验后归一化。精确并列时，以原始选项 ID 的字典序打破平局，预测并列使用相同规则。
- Test 中 gold 最高概率并列 35 题，预测最高概率并列 64 题。与“任一 gold 最大项均算正确”的评分口径会有差异。
- Soft Brier 是对所有候选的平方误差求和，再对决策取平均；Hard Brier 则与硬标签 one-hot 比较。Soft CE、KL 使用完整 gold 分布，Hard NLL、ECE 使用硬标签正确性。对数计算下限为 `1e-12`，不修改保存的预测概率。
- Score MAE 比较预测分布与 gold 分布的期望等级，保留原 rubric 的 0 起始等级值。
- 输入采用“每题独立因果上下文，同 case 五题批处理”。数据集公开榜单还可能采用一条上下文回答五题、不同 ECE/Brier 和并列规则，因此不直接横向比较数值。
- 训练样本诊断显示：同权重 FP32 的批量与无 padding 单题读出最大概率差小于 `4e-6`；BF16 为约 `0.03059`，检查样本的 argmax 均不变。批量布局或精度变化可能影响临界预测，所以后续比较应固定本协议。原生 DeltaNet 对带 padding 的 singleton 存在掩码限制，代码已明确拒绝该输入。

## 6. 产物与复现

正式运行目录：[完整产物](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/)。

- [运行配置](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/config.yaml)
- [Metadata 与模型／代码来源](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/metadata.json)
- [Dev 指标](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/dev/metrics.json) / [Test 指标](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/test/metrics.json)
- [Test 逐题预测](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/test/predictions.jsonl) / [Test 原始 logits](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/test/logits.npz)
- [产物复核记录](../outputs/01_baseline/b0/eval_20261021_170132_311730_seed42/verification.json)

已核对全部 2,600 条预测与原始数据的 case／问题／候选／gold 对应关系，NPZ 与 JSONL 的 logits、mask、身份一致。由 logits 重建概率的最大绝对误差小于 `1.4e-7`，保存的指标与复算结果完全一致。模型及代码哈希复核通过，没有 calibration 推理产物。

CPU 单元测试共 24 项通过，覆盖数据划分、指标公式、标签泄漏、候选映射、padding、超长输入拒绝、动态候选产物及 metadata 序列化。

在仓库根目录运行：

```bash
conda activate torch29-cu130
python experiments/01_baseline/run.py --prepare-only
CUDA_VISIBLE_DEVICES=1 python experiments/01_baseline/run.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

新的执行会生成独立 run ID，不覆盖本次结果。输入与运行细节见 [Baseline 说明](../experiments/01_baseline/README.md)。阶段 2 引用本报告的正式 run ID；不要把训练样本 smoke 结果作为 B0。
