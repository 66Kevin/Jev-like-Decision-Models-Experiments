# 阶段 4：温度校准

冻结模型，在独立 calibration split 上拟合温度，并评估校准前后的概率指标。本阶段尚未实现或运行。

后续引用模型 run ID 与保存的原始 logits，温度只在 calibration 上拟合。运行产物保存到 `outputs/04_calibration/<experiment_id>/<run_id>/`，不复制模型权重。
