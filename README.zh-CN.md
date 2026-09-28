# PhyST-MR：公开代码包

本代码包用于四级二尖瓣反流（MR）分类，包含生理监督引导的空间专家、使用 Kinetics-400 预训练权重的时空专家，以及类别特异性的概率融合；同时提供训练与推理代码。

**本仓库仅包含代码。**不包含超声心动图、患者标识、结构化测量值、模型 checkpoint 或预测文件。请自行提供有权使用的数据。本模型仅供研究使用，不是临床医疗器械。

## 方法流程

1. `scripts/train_spatial.py`：以 ImageNet 预训练权重初始化 ResNet18，进行**全量微调**，完成 study 级 MR 分类。
2. `scripts/train_spatial_physio.py`：从空间专家 checkpoint 出发，使用 10 项结构化超声心动图测量作为**仅在训练阶段使用的辅助回归目标**继续训练。推理时只需视频。
3. `scripts/train_temporal.py`：使用 torchvision 官方 Kinetics-400 权重初始化并训练 R(2+1)D-18。每段固定 16 帧视频逐帧复制为 32 个时间步。评价时，先对同一 study 的 clip logits 求平均，再计算 softmax。
4. `scripts/classwise_fusion.py`：在 Validation 上搜索四个空间专家权重，锁定后对 Test 评价一次。`scripts/apply_fusion.py` 可在没有真实标签的情况下应用锁定权重。

类别顺序为 `0=None/Trace, 1=Mild, 2=Moderate, 3=Severe`。空间专家先在每个 clip 内对逐帧特征求平均，再对最多 10 个 clip 做考虑有效 clip 掩码的均值汇聚。生理监督阶段的损失为 `balanced CE + 0.1 * masked SmoothL1`；推理时没有结构化数据输入。最终分类采用 study 级预测，而不是 clip 级预测。

## 环境配置

本代码已在 Python 3.10、PyTorch 2.5.1 和 torchvision 0.20.1 环境下测试。请先安装与 CUDA 环境匹配的 PyTorch 和 torchvision，然后运行：

```bash
pip install -r requirements.txt
```

可以运行 `python scripts/export_k400.py --output weights/r2plus1d_18_k400.pth`，获取并保存 torchvision 官方 `R2Plus1D_18_Weights.KINETICS400_V1` 权重；首次运行需要联网。第一阶段使用的 ImageNet ResNet18 权重由 torchvision 获取。训练预计使用 GPU；CPU 可以推理，但速度较慢。请只加载可信来源的 checkpoint。

## 数据接口

具体格式见 [DATA_FORMAT.md](DATA_FORMAT.md)。训练 manifest 只能包含 `train` 和 `val` 两种 split。患者级划分必须固定，且不同 split 之间不能有患者重叠。Validation 和 Test 分别使用独立的推理 manifest。脚本直接读取预先提取的 RGB 帧，**不负责**构建 E2 筛选后的队列，也不负责从 DICOM 抽帧。两个专家应使用相同的锁定帧数据和 clip 筛选结果。

## 训练

```bash
python scripts/train_spatial.py \
  --study-manifest data/studies_train_val.csv \
  --outdir runs/spatial_seed42 --seed 42 \
  --epochs 20 --batch-size 2

python scripts/train_spatial_physio.py \
  --study-manifest data/studies_train_val.csv \
  --structured-csv data/measurements_train_val.csv \
  --pretrained-checkpoint runs/spatial_seed42/best_softmax_model.pt \
  --outdir runs/spatial_physio_seed42 --seed 42 \
  --epochs 20 --batch-size 2 --phys-lambda 0.1

python scripts/train_temporal.py \
  --manifest data/clips_train_val.csv \
  --k400-checkpoint weights/r2plus1d_18_k400.pth \
  --outdir runs/temporal_seed42 --seed 42 \
  --epochs 8 --batch-size 4 --eval-batch-size 4 \
  --accumulate 3 --learning-rate 0.0002
```

空间专家的两个训练阶段均使用 AdamW、`1e-4` 权重衰减、batch size 2、FP16 自动混合精度和类别平衡交叉熵。第一阶段所有可训练层的学习率均为 `1e-4`。第二阶段的学习率分别为 `1e-5`（encoder）、`1e-4`（MR 分类头）和 `3e-4`（生理指标回归头）。时空专家使用 AdamW（学习率 `2e-4`、权重衰减 `0.0035`）、余弦退火、BF16 自动混合精度和梯度累积。各阶段均按 Validation Macro-F1 优先、QWK 次之、MAE 更低再次之的顺序选择 checkpoint。同一输出目录下的训练会从 `latest_checkpoint.pt` 续训；新实验请使用新目录。

## 推理

同一推理命令可生成 Validation 或 Test 概率。带生理监督的空间专家在推理时**不会读取结构化测量 CSV**。

```bash
python scripts/predict.py --expert spatial-physio \
  --manifest data/studies_val.csv \
  --checkpoint runs/spatial_physio_seed42/best_softmax_model.pt \
  --output runs/spatial_physio_seed42/val.csv

python scripts/predict.py --expert temporal \
  --manifest data/clips_val.csv \
  --checkpoint runs/temporal_seed42/best_native_macro_f1.pt \
  --output runs/temporal_seed42/val.csv
```

在 Test 上推理时，换用对应的独立 manifest 和输出路径。若要评价未加入生理监督的 ResNet18 全量微调基线，请使用 `--expert spatial` 及其对应的 checkpoint。

## 融合

```bash
python scripts/classwise_fusion.py \
  --resnet-val runs/spatial_physio_seed42/val.csv \
  --r2d-val runs/temporal_seed42/val.csv \
  --resnet-test runs/spatial_physio_seed42/test.csv \
  --r2d-test runs/temporal_seed42/test.csv \
  --outdir runs/fusion_seed42
```

融合公式为 `q_k = w_k * p_spatial,k + (1-w_k) * p_temporal,k`，随后归一化：`p_k = q_k / sum_j(q_j)`。每个 `w_k` 限制在 `[0.10, 0.90]`。代码先以 `0.05` 为步长粗搜，再在粗搜最优解的 `+/-0.10` 范围内以 `0.01` 为步长细搜。并列时依次比较 QWK、MAE，以及与四类等权重方案的距离。由于权重是在 Validation 上拟合的，**Validation 上的融合表现属于探索性结果**；只有锁定权重后才对留出的 Test 进行评价。生成的 `LOCKED_CLASSWISE_WEIGHTS.json` 用于后续应用：

```bash
python scripts/apply_fusion.py \
  --spatial spatial_probabilities.csv \
  --temporal temporal_probabilities.csv \
  --weights-json runs/fusion_seed42/LOCKED_CLASSWISE_WEIGHTS.json \
  --output fused_probabilities.csv
```

`apply_fusion.py` 不需要真实标签，也不需要结构化测量值。

## 公开范围

不要将受限数据、原始数据路径、SSH 密钥、checkpoint 或逐 study 预测结果提交到公开仓库。模型权重分发与数据集访问需要分别考虑授权和许可问题。目前仓库**未附代码许可证**。
