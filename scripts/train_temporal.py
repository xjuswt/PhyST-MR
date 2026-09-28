#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r2plus1d_18

LABELS = ["none_trace", "mild", "moderate", "severe"]
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(payload, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_flip(seed, epoch, sample_id):
    token = f"{seed}:{epoch}:{sample_id}".encode("ascii")
    return hashlib.sha256(token).digest()[0] < 128


class ClipDataset(Dataset):
    def __init__(self, frame, mode, seed, epoch, train):
        self.frame = frame.reset_index(drop=True)
        self.mode = mode
        self.seed = seed
        self.epoch = epoch
        self.train = train

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        paths = json.loads(row.frame_paths_json)
        if len(paths) != 16:
            raise ValueError(f"Clip {row.sample_id} must have exactly 16 frames")
        images = []
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        video = torch.stack(images, dim=1)  # C,T,H,W
        if self.mode == "L3":
            # The repository default is 32 frames. Deterministic repeat keeps exactly
            # the same fixed16 visual information without pixel interpolation.
            video = video.repeat_interleave(2, dim=1)
            video = (video - KINETICS_MEAN) / KINETICS_STD
        if self.train and deterministic_flip(self.seed, self.epoch, int(row.sample_id)):
            video = torch.flip(video, dims=[3])
        return video, int(row.label_id), int(row.study_id), int(row.sample_id)


def make_loader(frame, args, epoch, train):
    dataset = ClipDataset(frame, args.mode, args.seed, epoch, train)
    generator = torch.Generator().manual_seed(
        args.seed + epoch if train else args.seed + 100000
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.train_workers if train else args.eval_workers,
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )


def build_model(mode, k400_checkpoint):
    model = r2plus1d_18(weights=None, num_classes=4)
    if mode == "L3":
        state = torch.load(k400_checkpoint, map_location="cpu", weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing != ["fc.weight", "fc.bias"] or unexpected:
            raise RuntimeError(f"Unexpected K400 load: missing={missing}, unexpected={unexpected}")
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.fc.bias)
    return model


def ordinal_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    error = np.abs(y - pred)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(4), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(4), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=np.arange(4), average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "mae_grade": float(error.mean()),
        "one_off_accuracy": float((error <= 1).mean()),
        "two_or_more_off": float((error >= 2).mean()),
        "per_class": {
            LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(per_f1[i]), "support": int(support[i])}
            for i in range(4)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(4)).tolist(),
    }


def binary_metrics(y, probability, threshold_class):
    target = (np.asarray(y) >= threshold_class).astype(int)
    score = probability[:, threshold_class:].sum(axis=1)
    pred = (score >= 0.5).astype(int)
    specificity = recall_score(target, pred, pos_label=0, zero_division=0)
    return {
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
        "f1": float(f1_score(target, pred, zero_division=0)),
        "sensitivity": float(recall_score(target, pred, zero_division=0)),
        "specificity": float(specificity),
        "threshold": 0.5,
    }


def choose_thresholds(y, scores):
    candidates = np.unique(np.r_[np.arange(0.25, 3.0, 0.25),
                                 np.quantile(scores, np.linspace(0.05, 0.95, 17))])
    candidates = candidates[(candidates > 0) & (candidates < 3)]
    best = None
    for t1 in candidates:
        for t2 in candidates[candidates > t1]:
            for t3 in candidates[candidates > t2]:
                metrics = ordinal_metrics(y, np.digitize(scores, [t1, t2, t3]))
                rank = (metrics["macro_f1"], metrics["qwk"], -metrics["mae_grade"])
                if best is None or rank > best[0]:
                    best = rank, [float(t1), float(t2), float(t3)], metrics
    return best[1], best[2]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    losses = []
    for video, label, study, sample in loader:
        video = video.to(device, non_blocking=True)
        label_gpu = label.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(video)
            loss = torch.nn.functional.cross_entropy(logits, label_gpu)
        for i in range(len(label)):
            rows.append((int(study[i]), int(sample[i]), int(label[i]), *logits[i].float().cpu().tolist()))
        losses.append(float(loss))
    clip = pd.DataFrame(rows, columns=["study_id", "sample_id", "label_id", "logit_0", "logit_1", "logit_2", "logit_3"])
    grouped = clip.groupby("study_id", sort=True)
    study = grouped[[f"logit_{i}" for i in range(4)]].mean()
    labels = grouped.label_id.first().to_numpy(dtype=int)
    if not (grouped.label_id.nunique() == 1).all():
        raise RuntimeError("Inconsistent labels within study")
    probability = torch.softmax(torch.tensor(study.to_numpy()), dim=1).numpy()
    metrics = ordinal_metrics(labels, probability.argmax(axis=1))
    metrics["moderate_or_greater"] = binary_metrics(labels, probability, 2)
    metrics["severe"] = binary_metrics(labels, probability, 3)
    return clip, study.index.to_numpy(), labels, probability, float(np.mean(losses)), metrics


def save_eval(outdir, name, result, thresholds=None):
    clip, studies, labels, probability, loss, metrics = result
    scores = probability @ np.arange(4)
    table = pd.DataFrame({"study_id": studies, "label_id": labels,
                          "pred_softmax": probability.argmax(axis=1), "expected_grade": scores})
    for i, label in enumerate(LABELS):
        table[f"prob_{label}"] = probability[:, i]
    if thresholds is not None:
        table["pred_expected_val_thresholds"] = np.digitize(scores, thresholds)
    clip.to_csv(outdir / f"{name}_clip_logits.csv", index=False)
    table.to_csv(outdir / f"{name}_study_predictions.csv", index=False)
    payload = {"loss": loss, "native_softmax": metrics}
    if thresholds is not None:
        payload["expected_grade_exploratory"] = {
            "thresholds": thresholds,
            "metrics": ordinal_metrics(labels, np.digitize(scores, thresholds)),
        }
    (outdir / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["L3"], default="L3")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k400-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-clips-per-class", type=int, default=2)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    frame = pd.read_csv(args.manifest)
    required = {"study_id", "sample_id", "label_id", "split", "frame_paths_json"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Clip manifest missing: {sorted(required - set(frame.columns))}")
    if frame.sample_id.duplicated().any():
        raise ValueError("Clip manifest has duplicate sample IDs")
    if not frame.split.isin(["train", "val"]).all():
        raise ValueError("Training manifest must contain only train and val clips")
    train = frame[frame.split.eq("train")].copy()
    val = frame[frame.split.eq("val")].copy()
    if args.smoke:
        train = train.groupby("label_id", group_keys=False).head(args.smoke_clips_per_class)
        val = val.groupby("label_id", group_keys=False).head(args.smoke_clips_per_class)
        args.epochs = 1
        args.train_workers = args.eval_workers = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for temporal expert training")
    device = torch.device("cuda")
    model = build_model(args.mode, args.k400_checkpoint).to(device)
    learning_rate = args.learning_rate if args.learning_rate is not None else 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0035)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = args.outdir / "latest_checkpoint.pt"
    best_path = args.outdir / "best_native_macro_f1.pt"
    start_epoch, history, best_rank, patience = 1, [], None, 0
    if checkpoint.exists() and not args.smoke:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history, best_rank, patience = state["history"], tuple(state["best_rank"]), state.get("patience", 0)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loader = make_loader(train, args, epoch, True)
        optimizer.zero_grad(set_to_none=True)
        running = []
        epoch_started = time.time()
        for step, (video, label, _, _) in enumerate(loader, 1):
            video, label = video.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(video)
                loss = torch.nn.functional.cross_entropy(logits, label)
                scaled = loss / args.accumulate
            scaled.backward()
            if step % args.accumulate == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running.append(float(loss))
            if step % 200 == 0:
                elapsed = time.time() - epoch_started
                log(f"mode={args.mode} epoch={epoch}/{args.epochs} train_step={step}/{len(loader)} "
                    f"clips={min(step * args.batch_size, len(train))}/{len(train)} "
                    f"mean_loss={np.mean(running[-200:]):.5f} clips_per_sec="
                    f"{min(step * args.batch_size, len(train)) / max(elapsed, 1):.2f}")
        result = evaluate(model, make_loader(val, args, epoch, False), device)
        metrics = result[-1]
        rank = (metrics["macro_f1"], metrics["qwk"], -metrics["mae_grade"])
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank, patience = rank, 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, "rank": rank}, best_path)
        else:
            patience += 1
        record = {"epoch": epoch, "train_loss": float(np.mean(running)), "val_loss": result[-2],
                  "val_native": metrics, "improved": improved,
                  "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(record)
        scheduler.step()
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "scheduler": scheduler.state_dict(),
                     "history": history, "best_rank": best_rank, "patience": patience}, checkpoint)
        (args.outdir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        log(f"mode={args.mode} epoch={epoch}/{args.epochs} train_loss={record['train_loss']:.5f} "
            f"val_loss={record['val_loss']:.5f} macro_f1={metrics['macro_f1']:.5f} qwk={metrics['qwk']:.5f} improved={improved}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    final = evaluate(model, make_loader(val, args, 0, False), device)
    thresholds, _ = choose_thresholds(final[2], final[3] @ np.arange(4))
    save_eval(args.outdir, "validation", final, thresholds)
    info = {"mode": args.mode, "seed": args.seed, "epochs_requested": args.epochs,
            "best_epoch": best["epoch"], "effective_batch": args.batch_size * args.accumulate,
            "learning_rate": learning_rate,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "manifest_sha256": sha256(args.manifest), "test_accessed": False,
            "trainer_sha256": sha256(Path(__file__)),
            "status": "SMOKE_COMPLETED" if args.smoke else "COMPLETED"}
    (args.outdir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (args.outdir / ("SMOKE_COMPLETED" if args.smoke else "COMPLETED")).touch()
    log(json.dumps(info))


if __name__ == "__main__":
    main()
