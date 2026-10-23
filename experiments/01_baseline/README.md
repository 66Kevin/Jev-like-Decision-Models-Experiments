# 阶段 1：B0 原始模型基线

B0 测量本地 Qwen3.5-0.8B 在没有专项训练时的决策能力，后续阶段 2 直接引用这份结果。

## 固定协议

- 数据固定到配置中的 Hugging Face revision；加载前校验八个原始文件的 SHA256。
- 整个 case 划入 train / dev / calibration / test = 960 / 120 / 120 / 400；同一状态的五道题不拆开。
- 每道题独立输入原始 state 的规范 JSON、原始题目说明、类型及候选项。候选沿用原始顺序，映射为 `A. option_id: option_text`。
- 使用固定英文分类提示和模型官方 chat template，关闭 thinking。gold、hard label、factors 等不进入提示。
- 校验每个字母既是单 token，又是完整提示后的单 token continuation。
- 模型 `eval()`、冻结权重、`inference_mode()`，BF16、SDPA，关闭 KV cache；无训练、无生成 token。
- 每个 case 的五道题组成一批，左侧 padding，用 attention mask 计算 position IDs。原生 LM Head 仅投影最后一个位置（`logits_to_keep=1`），取有效字母 logits 做 FP32 softmax，温度为 1。
- 上限 8192 token，超过立即报错，不截断、不跳过；正式评估之前检查所有待评估输入。
- 三个 train case 用于预热和运行检查；首个 case 额外比较批量与逐题读出，记录 BF16 数值误差。它们均不参与正式指标。当前运行验证中，BF16 批量与逐题最大 logits 差为 0.25、概率差约 0.0306；将同一检查点权重转换到 FP32 后，相同对照的差异小于 0.00002 logits / 0.000004 概率，说明本次差异来自数值计算。BF16 检查使用 0.5 logits / 0.05 概率的经验容差；临界样本的 argmax 仍可能受精度影响。
- 原生 Qwen3.5 对单行 padding 的 DeltaNet mask 有特殊限制；readout 明确拒绝有 padding 的单行输入。正式五题 batch 不受此限制，单题检查直接使用无 padding 的输入。
- 正式运行同时评估 dev 与官方 test；不在 test 上调整提示或超参数，不对 calibration 做推理。

## 运行

在仓库根目录执行。需要可用的 PyTorch、Transformers（支持 `Qwen3_5ForCausalLM`）、NumPy、PyArrow、PyYAML；当前服务器环境为 `/home/wangyueyi/miniconda3/envs/torch29-cu130/bin/python`。

```bash
# 纯数据检查与划分，不加载模型、不访问 GPU
python experiments/01_baseline/run.py --prepare-only

# 仅三个训练 case 的运行检查，输出名以 smoke_train_ 开头
CUDA_VISIBLE_DEVICES=1 python experiments/01_baseline/run.py --smoke

# 正式 B0：120 个 dev case + 400 个 test case
CUDA_VISIBLE_DEVICES=1 python experiments/01_baseline/run.py

# CPU 单元测试
PYTHONPATH=src python -m unittest discover -s tests -v
```

`CUDA_VISIBLE_DEVICES=1` 将物理 GPU 1 映射到配置中的 `cuda:0`。实际设备、环境版本、可选内核是否可用均写入 metadata。缺少 FLA / causal-conv1d 时，Transformers 使用原生 PyTorch 路径，延迟应按实际运行环境解读。

原始数据不纳入 Git。全新检出时，可通过已记录在 `data/splits/.../manifest.json` 的 `source_manifest` 重建原始文件，再运行 `--prepare-only`。以下命令明确联网，只下载锁定版本并校验内容：

```bash
python - <<'PY'
import hashlib, json, pathlib, urllib.request
import yaml
config = yaml.safe_load(pathlib.Path('experiments/01_baseline/configs/b0.yaml').read_text())
splits = pathlib.Path(config['dataset']['split_dir'])
manifest = json.loads((splits / 'manifest.json').read_text())['source_manifest']
root = pathlib.Path(config['dataset']['raw_dir'])
for item in manifest['files']:
    path = root / item['file']
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(item['url'], timeout=120).read()
    assert hashlib.sha256(data).hexdigest() == item['sha256'], path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
(root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
```

## 结果留存与解释

每次运行创建独立目录 `outputs/01_baseline/b0/{eval|smoke_train}_北京时间_seed42/`，不覆盖已有结果。

```text
run/
├── config.yaml                 # 本次完整配置
├── metadata.json               # running / complete / failed；设备、版本、文件哈希、协议
├── run.log
├── source/                     # 本次共享代码、runner、测试和说明快照
├── splits/                     # 数据划分与来源清单快照
├── dev/                        # smoke 模式则只有 smoke_train/
│   ├── predictions.jsonl       # 原始候选身份、gold、硬标签、预测概率、logits
│   ├── logits.npz              # 对齐 case/question/option IDs 与有效候选 mask
│   ├── metrics.json            # 总体、工作流、题型、题目指标与 case bootstrap 区间
│   └── timing.json             # 每个 case 的计时和输入长度，p50 / p95
└── test/
```

Accuracy 对照 gold 的 argmax，精确并列时按原始选项 ID 排序打破平局。它表示与数据集教师硬标签的一致性。概率指标记录 Hard NLL、Soft CE、KL(q‖p)、硬／软 Brier、15 等宽分箱 ECE，以及 Score 期望分数 MAE；均为未经温度校准的结果。

延迟按一个 case 的五道独立问题统计，包括分词、数据传输、同步推理及概率搬回 CPU，排除模型加载、预热、输入预检、指标计算与文件写入。显存为 PyTorch allocator 的峰值，包含模型本身；不等同于整张显卡的占用。

只有 metadata 中 `status=complete` 且 mode 为 `eval` 的运行才可作为正式 B0。失败运行会保留已产生的结果及错误，便于追溯；`smoke_train` 只证明流程可运行。
