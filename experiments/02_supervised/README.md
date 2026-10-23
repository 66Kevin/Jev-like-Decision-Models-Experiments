# 阶段 2：专项训练

本阶段仅使用 `argmax(gold)` 硬标签，研究 Backbone 适配（Q1）和 Head 架构（Q2）。

计划实验为G1、G2、P0、P1、P2、C1；B0引用 `../01_baseline/`。G1／G2／P0／P1已完成训练、选模、test和离线复核；正式状态以各run的 `metadata.json` 为准。P2／C1尚未启动。详细矩阵见根目录README。

配置放在 `configs/`，运行结果统一保存到 `outputs/02_supervised/<experiment_id>/<run_id>/`。复用根目录 `data/splits/` 的划分，避免各组自行切分数据。

## G1：保留 LM Head，仅训练 Backbone LoRA

使用与 B0 完全相同的模型、提示、候选字母映射和数据划分。原模型全部权重（包含绑定的输入 embedding 与 LM Head）冻结，仅训练 186 个投影模块上的 FP32 LoRA A/B 参数。标签是 `argmax(gold)`；损失是有效候选字母上的硬标签交叉熵。

固定配置见 [configs/g1.yaml](configs/g1.yaml)：rank 16、alpha 32、dropout 0，AdamW 学习率 2e-4，3 个 epoch，每步 32 道题（microbatch 8 × 累积 4），5% warmup（向上取整 23 步）后 cosine，种子 42/43/44。每个种子共 450 个优化步。

```bash
G1_PYTHON=/home/wangyueyi/miniconda3/envs/torch29-cu130/bin/python
$G1_PYTHON experiments/02_supervised/run.py --prepare-only
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G1_PYTHON experiments/02_supervised/run.py --smoke
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G1_PYTHON experiments/02_supervised/run.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G1_PYTHON experiments/02_supervised/run.py --resume outputs/02_supervised/g1/<run_id>
```

`--prepare-only` 校验固定数据与 B0 来源，无需 GPU。`--smoke` 只用 train 做两步优化、原始权重校验与 adapter 保存／重载检查，不推理 dev/test；它保存在独立 `smoke_train_*` 目录。正式运行依次训练全部三个种子，每个 epoch 使用原生 B0 后端评估 dev，以最低 hard NLL 选 checkpoint（包含 step 0，平局选更早一步）。所有种子的选择先写入锁定的 `selections.json`，然后才评估官方 test。

训练可使用项目内 `.runtime/fla` 的 DeltaNet chunk 内核，原生 normalization／convolution 保留；dev/test 恢复原生 B0 内核。训练内核已经过 train-only 数值与梯度检查，证据文件 `kernel_profile_train.json` 和哈希随运行保存。验证／测试延迟包含未合并 LoRA 的开销，不代表合并部署后的性能。训练峰值显存和原生评估显存分别记录，评估时优化器状态移到 CPU。

每个运行目录包含：

- `config.yaml`、`source/`、`splits/`、`metadata.json`、`run.log`、`pid`：来源、状态与复现信息。
- `seed42/` 等：`train.jsonl`、`model.json`、`tokenization.json`、`training_memory.json`、`training_backend.json`。
- 每个种子的 `checkpoints/step*/`：未合并 adapter、同一步优化器／scheduler／Python、NumPy、Torch、CUDA RNG 状态。`latest.json` 原子更新，只在完整 checkpoint 保存后指向它；保留初始、各 epoch 和最新 checkpoint。
- 每个种子的 `dev_epoch0/`：验证来源一致、zero-adapter 训练样本预测一致后复用的 B0 dev 结果；`dev_epoch1/` 至 `dev_epoch3/`：逐轮预测、logits、指标和原生评估显存。
- `selections.json`、每个种子的 `test/`、`summary.json`：锁定选择、官方测试结果、三个种子的均值／样本标准差，以及相对 B0 的配对 case bootstrap Accuracy 差值。

`--resume` 要求配置、代码快照、模型来源、Python／依赖构建、本地 FLA 源码及预检报告哈希均一致，在写入日志或 PID 前完成核对，只从完整优化步恢复；已有完整 dev/test 结果会保留。若 held-out 推理留下不完整目录，程序会明确停止，避免静默重复测试。不要同时启动同一运行目录的两个进程。运行中／失败不等于实验完成；以 `metadata.json` 的状态和已存在结果为准。

当前工作区已具备 Conda 依赖、`.runtime/fla` 与训练样本预检证据。全新环境需要先准备固定数据、B0 产物并重新完成训练内核预检；仅安装依赖不等于已经通过预检。本项目的 FLA 包单独安装到工作区，不改动共享环境：

```bash
$G1_PYTHON -m pip install --no-deps --target .runtime/fla fla-core==0.5.2
```

正式运行与结果状态见 [G1 记录](../../reports/02_supervised_g1.md)。分工作流、分题型指标保存在各 seed 的 `test/metrics.json`；`summary.json` 汇总总体指标的均值／标准差与配对 Accuracy 差值。

## G2：完整微调 Backbone 与原有 LM Head

入口是 [run_g2.py](run_g2.py)，固定配置见 [configs/g2.yaml](configs/g2.yaml)。从原始模型重新开始，全部 752,393,024 个文本参数参与训练，包含绑定的输入 embedding 与 LM Head。使用 FP32 主权重、梯度及 AdamW 状态，CUDA BF16 autocast；原生 FP32 rotary buffers 保留。评估通过独立的原生 BF16 参数副本进行，避免把训练主权重反复转换而丢失精度。

G2 与 G1 共用硬标签、提示、划分、每步 32 题、3 个 epoch 和三个种子。G2 学习率为 2e-5，使用 fused AdamW；训练样本预检选择 microbatch 4 × 累积 8。学习率与训练参数精度属于各组固定配方，因此比较反映这两套配方的结果，不能直接视为各方法的最优上限。

```bash
G2_PYTHON=/home/wangyueyi/miniconda3/envs/torch29-cu130/bin/python
$G2_PYTHON experiments/02_supervised/run_g2.py --prepare-only
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G2_PYTHON experiments/02_supervised/run_g2.py --smoke
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G2_PYTHON experiments/02_supervised/run_g2.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$PWD/.runtime/triton" $G2_PYTHON experiments/02_supervised/run_g2.py --resume outputs/02_supervised/g2/<run_id>
```

`--smoke` 只用 train：优化两步，完整保存并重载 FP32 权重、Adam／scheduler／RNG，验证原生评估 logits 一致，再执行一步真实优化来验证 fused Adam 恢复。正式运行在第 2 步、之后每 50 步及每个 epoch 保存完整检查点。每代包含原生模型权重、trainer 状态和校验 manifest；完成后才原子更新 `latest.json`，保留各 epoch 和最新一代。step 0 保存原始模型引用与哈希，避免重复保存未训练权重。恢复会归档检查点后尚未提交的训练日志，再按相同题目顺序重放。

每轮 dev 使用 CPU 上的 FP32 主模型创建独立评估副本，结束后验证主权重哈希与参数身份不变，再恢复优化器设备。所有种子按 dev hard NLL 选择并锁定后才评估 test；不使用 calibration。`summary.json` 汇总三个种子的均值、样本标准差和相对 B0／G1 的配对 case bootstrap Accuracy 差值。G1 与 G2 的比较使用各自种子的平均正确率，不把概率平均当成集成模型。运行状态以 `metadata.json` 为准，运行入口不会自动启动其他实验组。

### G2 完成后的离线复核

本次恢复控制器已接续完整流程：训练／dev 选模／test 全部完成后，自动执行 [finish_g2.py](finish_g2.py) 并更新报告和根目录 README。当前状态查看正式运行目录的 `metadata.json`，恢复及自动收尾状态查看 `recovery/allocator_20261022/continuation_status.json`。

如需在完成后重新核对已有结果：

```bash
$G2_PYTHON experiments/02_supervised/finish_g2.py \
  --run outputs/02_supervised/g2/train_20261022_104739_026131 --update-docs
```

该命令只用 CPU 读取保存产物，核对预测／logits、指标、三个 seed 的选模和完整模型校验，不重新进行模型推理。训练或 test 未完成时会拒绝生成最终结论。复核通过后生成 `analysis.json`、`verification.json`、`selected_models.json`，并更新 [G2 报告](../../reports/02_supervised_g2.md) 和曲线。

## P0：冻结原始 Backbone，只训练 Pointer Head

使用 [p0.yaml](configs/p0.yaml) 和 [run_p0.py](run_p0.py)。Pointer 取最后输入位置与候选文本末尾的表示，训练两个共享的 128 维投影，共 262,144 个 FP32 参数；原始模型保持冻结，完整输入与 B0 一致。

```bash
P0_PYTHON=/home/wangyueyi/miniconda3/envs/torch29-cu130/bin/python
$P0_PYTHON experiments/02_supervised/run_p0.py --prepare-only
CUDA_VISIBLE_DEVICES=0 $P0_PYTHON experiments/02_supervised/run_p0.py --smoke
CUDA_VISIBLE_DEVICES=0 $P0_PYTHON experiments/02_supervised/run_p0.py
CUDA_VISIBLE_DEVICES=0 $P0_PYTHON experiments/02_supervised/run_p0.py --resume outputs/02_supervised/p0/<run_id>
```

正式运行自动提取 train／dev 的冻结特征，分别训练 seed 42／43／44 的随机 Head；epoch0 实测随机 Head，不能复制 B0 指标。全部选模锁定后才分别执行未缓存的完整 test，延迟计入 Backbone 和位置映射。校准集不使用。运行记录及固定配方见 [P0 报告](../../reports/02_supervised_p0.md)。

本次 P0 run 为 `train_20261022_232846_262432`，三个 seed 均按 dev NLL 选中 epoch3／step450。Test Accuracy 为 **63.80% ± 0.90 pp**，较 B0 提升 17.50 pp；未经温度校准。它尚不能单独证明 Pointer 优于 LM Head，仍需匹配 Backbone 适配方式的对照。

完成运行后，可仅用 CPU 复核已有产物，不重新执行模型推理：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $P0_PYTHON experiments/02_supervised/finish_p0.py \
  --run outputs/02_supervised/p0/train_20261022_232846_262432
```

复核生成 `analysis.json`、`verification.json`、`selected_heads.json` 和报告用图。`selected_heads.json` 记录三个 dev 所选 Head 及原始 Backbone 引用；Head 不是完整模型，复用时仍需原始 Backbone、同一输入及候选位置映射。已经完成的 run 不需要再调用 `--resume`。缓存 dev 不报告完整推理延迟，正式 test 没有特征缓存。

## P1：Pointer Head 与 Backbone LoRA 联合训练

入口 [run_p1.py](run_p1.py)，配置 [p1.yaml](configs/p1.yaml)，记录见 [P1 报告](../../reports/02_supervised_p1.md)。保持 P0 的 128 维 Pointer Head，增加 G1 的 r16／alpha32、186 个目标投影 LoRA，联合训练 **11,084,800** 个 FP32 参数。Head 按相同 seed 独立随机初始化，与 P0 起点一致；不加载 P0 的训练权重。

LoRA LR 为 2e-4，Head LR 为 1e-3；有效 batch32 = micro8×累积4，seed42／43／44，各3 epochs／450步。只缓存 tokens 和候选位置，每次更新重新计算 Backbone；FP32 Head 保留 query/key 梯度。随机 Head＋zero-LoRA 的 epoch0 逐 seed 实测，所有选择锁定后才进行完整 test。

以下命令在 **26服务器** 的仓库根目录执行，使用专用 Conda 环境和物理 GPU0 RTX4090：

```bash
P1_PYTHON="$PWD/.runtime/conda/p1_26/bin/python"
$P1_PYTHON experiments/02_supervised/run_p1.py --prepare-only
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRITON_CACHE_DIR="$PWD/.runtime/triton_p1_26"
$P1_PYTHON experiments/02_supervised/run_p1.py --smoke
$P1_PYTHON experiments/02_supervised/run_p1.py
$P1_PYTHON experiments/02_supervised/run_p1.py --resume outputs/02_supervised/p1/<run_id>
```

运行入口校验 `alg26`、GPU映射与实际UUID。训练使用 FLA chunk，dev/test 使用未合并 LoRA 的原生模型；原始 BF16 参数和 FP32 buffers 保持冻结。检查点将同一步的 adapter、Head、optimizer／scheduler／RNG 一起保存并校验，恢复不允许混用不同步骤。部分 heldout 结果会拒绝隐式重跑。

26的train-only实卡预检及GPU smoke已通过；15项新CPU测试已在26环境通过。正式run `train_20261023_102132_592199` 已于北京时间2026-10-23 11:03完成，三个seed均按dev NLL选中epoch2／step300。Test Accuracy为 **76.30% ± 0.58 pp**，相对P0提高12.50 pp，相对G1的差区间包含0。旧组在5090、P1在4090，延迟不能直接作为架构差异比较。

### P1 完成后的离线复核

在26服务器相同Conda环境运行以下命令，读取已有预测／联合检查点并生成分析与图表，不执行模型推理，也不改写原summary或人工报告：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $P1_PYTHON experiments/02_supervised/finish_p1.py \
  --run outputs/02_supervised/p1/train_20261023_102132_592199
```

本次复核已通过：13,200条P1预测、20,000条参考test预测、12份联合epoch检查点、34份归档源码均一致，新增9项离线测试通过。复核生成 `analysis.json`、`verification.json` 和 `selected_joint_models.json`，四组PNG／SVG图及[最终审查记录](../../outputs/02_supervised/p1/train_20261023_102132_592199/final_review.json)已归档。模型注册表引用原始冻结Backbone与同一步的Head＋adapter；它不是完整合并模型。已完成run不需要再调用 `--resume`，后续实验单独建run。
