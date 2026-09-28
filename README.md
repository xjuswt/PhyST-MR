# PhyST-MR: public code package

[简体中文](README.zh-CN.md)

Training and inference code for four-grade mitral regurgitation (MR) classification with a physiology-guided spatial expert, a Kinetics-400 pretrained spatiotemporal expert, and class-wise probability fusion.

**Code only.** This repository does not include echocardiograms, patient identifiers, structured measurements, checkpoints, or prediction files. Supply data that you are authorized to use. The model is a research system, not a clinical device.

## Pipeline

1. `scripts/train_spatial.py`: ImageNet-initialized **full** fine-tuning of ResNet18 for study-level MR classification.
2. `scripts/train_spatial_physio.py`: continue from the spatial checkpoint with ten structured echocardiographic measurements as **training-only auxiliary regression targets**. Inference is video-only.
3. `scripts/train_temporal.py`: train R(2+1)D-18 initialized from the official torchvision Kinetics-400 weights. Each fixed 16-frame clip is repeated frame-by-frame to 32 temporal steps. At evaluation, average clip logits within a study, then softmax.
4. `scripts/classwise_fusion.py`: choose four spatial weights on Validation, lock them, and evaluate Test once. `scripts/apply_fusion.py` applies locked weights without labels.

Class order is `0=None/Trace, 1=Mild, 2=Moderate, 3=Severe`. The spatial expert averages frame features within each clip, then uses a mask-aware mean over up to ten clips. The physiology loss is `balanced CE + 0.1 * masked SmoothL1`; there is no structured input at inference. All four models use study-level predictions, not clip-level scoring, for final classification.

## Setup

Tested with Python 3.10, PyTorch 2.5.1, and torchvision 0.20.1. Install the matching PyTorch/torchvision builds for your CUDA runtime, then:

```bash
pip install -r requirements.txt
```

Obtain the official torchvision `R2Plus1D_18_Weights.KINETICS400_V1` state dict with `python scripts/export_k400.py --output weights/r2plus1d_18_k400.pth` (requires network access once). The ImageNet ResNet18 weights used by the first stage are resolved through torchvision. GPU execution is expected for training; CPU inference is supported but slow. Load only checkpoints you trust.

## Data interface

See [DATA_FORMAT.md](DATA_FORMAT.md). The training manifests must contain only `train` and `val` rows. Keep the patient-level split fixed and free of overlap. Validation and Test are separate inference manifests. The scripts consume pre-extracted RGB frames and do not create the E2-selected cohort or perform DICOM extraction; provide the same locked frames and clip selection to both experts.

## Train

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

The spatial stages use AdamW with weight decay `1e-4`, batch size 2, FP16 autocast, and balanced cross-entropy. Stage 1 uses LR `1e-4` for all trainable layers. Stage 2 uses LRs `1e-5` (encoder), `1e-4` (MR classifier), and `3e-4` (physiology head). The temporal stage uses AdamW (`2e-4`, weight decay `0.0035`), cosine annealing, BF16 autocast, and gradient accumulation. Each stage selects a checkpoint by Validation macro-F1, then QWK, then lower MAE. Training resumes from `latest_checkpoint.pt` within the same output directory; use a new directory for a new run.

## Predict

The same inference command can generate Validation or Test probabilities. Spatial physiology inference **does not read the measurements CSV**.

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

Repeat for Test with separate manifests and output paths. To evaluate the unmodified full-fine-tuned spatial baseline, use `--expert spatial` and its own checkpoint.

## Fuse

```bash
python scripts/classwise_fusion.py \
  --resnet-val runs/spatial_physio_seed42/val.csv \
  --r2d-val runs/temporal_seed42/val.csv \
  --resnet-test runs/spatial_physio_seed42/test.csv \
  --r2d-test runs/temporal_seed42/test.csv \
  --outdir runs/fusion_seed42
```

The fusion formula is `q_k = w_k * p_spatial,k + (1-w_k) * p_temporal,k`, followed by `p_k = q_k / sum_j(q_j)`. Each `w_k` is restricted to `[0.10, 0.90]`. The code searches a 0.05 coarse grid, then a 0.01 fine grid within `+/-0.10` of the coarse optimum. Ties are resolved by QWK, MAE, then distance from equal weights. These Validation-fitted weights are exploratory with respect to Validation performance; the held-out Test is evaluated only after the weights are locked. The generated `LOCKED_CLASSWISE_WEIGHTS.json` is the deployment input:

```bash
python scripts/apply_fusion.py \
  --spatial spatial_probabilities.csv \
  --temporal temporal_probabilities.csv \
  --weights-json runs/fusion_seed42/LOCKED_CLASSWISE_WEIGHTS.json \
  --output fused_probabilities.csv
```

`apply_fusion.py` does not need true labels or structured measurements.
