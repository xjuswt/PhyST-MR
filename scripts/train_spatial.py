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
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABELS = ["none_trace", "mild", "moderate", "severe"]
PROB_COLS = [f"prob_{label}" for label in LABELS]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def deterministic_flip(seed, epoch, study_id):
    value = f"{seed}:{epoch}:{study_id}".encode("ascii")
    return hashlib.sha256(value).digest()[0] < 128


class PairedStudyFrameDataset(Dataset):
    def __init__(self, study_df, seed, epoch, train):
        self.df = study_df.reset_index(drop=True)
        self.seed = seed
        self.epoch = epoch
        self.train = train
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return self.norm(torch.from_numpy(array).permute(2, 0, 1))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        clips = json.loads(row.clip_frame_paths)[:10]
        if not clips or any(not paths for paths in clips):
            raise ValueError(f"Study {row.study_id_int} has an empty clip")
        tensors = []
        for paths in clips:
            if len(paths) >= 16:
                chosen = np.linspace(0, len(paths) - 1, 16).round().astype(int)
                paths = [paths[position] for position in chosen]
            else:
                paths = paths + [paths[-1]] * (16 - len(paths))
            tensors.append(torch.stack([self.load_image(path) for path in paths]))
        x = torch.stack(tensors)
        if self.train and deterministic_flip(
            self.seed, self.epoch, int(row.study_id_int)
        ):
            x = torch.flip(x, dims=[4])
        return x, int(row.label_id), int(row.study_id_int)


def collate_studies(batch):
    max_clips = max(item[0].shape[0] for item in batch)
    frames, channels, height, width = batch[0][0].shape[1:]
    xs, masks, labels, studies = [], [], [], []
    for x, label, study in batch:
        clips = x.shape[0]
        padded = torch.zeros(
            max_clips, frames, channels, height, width, dtype=x.dtype
        )
        padded[:clips] = x
        mask = torch.zeros(max_clips, dtype=torch.float32)
        mask[:clips] = 1.0
        xs.append(padded)
        masks.append(mask)
        labels.append(label)
        studies.append(study)
    return (
        torch.stack(xs),
        torch.stack(masks),
        torch.tensor(labels),
        torch.tensor(studies),
    )


class MeanTemporal(nn.Module):
    output_dim = 512

    def forward(self, sequence):
        return sequence.mean(dim=1)


class AttentionTemporal(nn.Module):
    output_dim = 512

    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, sequence):
        weights = torch.softmax(self.score(sequence), dim=1)
        return (sequence * weights).sum(dim=1)


class ResidualTCNTemporal(nn.Module):
    output_dim = 512

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(128, 512, kernel_size=3, padding=1)

    def forward(self, sequence):
        x = sequence.transpose(1, 2)
        x = x + self.conv2(self.activation(self.conv1(x)))
        return x.mean(dim=2)


class GRUTemporal(nn.Module):
    output_dim = 512

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.projection = nn.Linear(256, 512)

    def forward(self, sequence):
        hidden_sequence, _ = self.gru(sequence)
        return self.projection(hidden_sequence.mean(dim=1))


TEMPORAL_MODULES = {
    "mean": MeanTemporal,
    "attention": AttentionTemporal,
    "tcn": ResidualTCNTemporal,
    "gru": GRUTemporal,
}


class TemporalV1b(nn.Module):
    def __init__(self, variant, seed, imagenet=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
        base = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 2000)
            self.temporal = TEMPORAL_MODULES[variant]()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            self.classifier = nn.Linear(512, 4)

    def forward(self, x, clip_mask):
        batch, clips, frames, channels, height, width = x.shape
        features = self.encoder(
            x.reshape(batch * clips * frames, channels, height, width)
        ).flatten(1)
        features = features.view(batch * clips, frames, 512)
        clip_features = self.temporal(features).view(batch, clips, 512)
        study_features = (
            (clip_features * clip_mask[:, :, None]).sum(dim=1)
            / clip_mask.sum(dim=1).clamp_min(1.0).view(batch, 1)
        )
        return self.classifier(study_features)


def classification_metrics(y, prediction):
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    difference = np.abs(y - prediction)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=range(4), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, prediction, labels=range(4), average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y, prediction, weights="quadratic")),
        "mae_grade": float(difference.mean()),
        "one_off_accuracy": float((difference <= 1).mean()),
        "two_or_more_off": float((difference >= 2).mean()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=range(4)).tolist(),
    }


def choose_thresholds(y, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 17)))
    candidates = candidates[(candidates > 0.0) & (candidates < 3.0)]
    candidates = np.unique(np.r_[np.arange(0.25, 3.0, 0.25), candidates])
    best = None
    for first in candidates:
        for second in candidates:
            if second <= first:
                continue
            for third in candidates:
                if third <= second:
                    continue
                thresholds = [float(first), float(second), float(third)]
                prediction = np.digitize(scores, thresholds)
                result = classification_metrics(y, prediction)
                rank = (
                    result["macro_f1"],
                    result["qwk"],
                    -result["mae_grade"],
                    result["accuracy"],
                )
                if best is None or rank > best[0]:
                    best = rank, thresholds, result
    return best[1], best[2]


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    probabilities, labels, studies, losses = [], [], [], []
    for x, mask, y, study_id in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x, mask)
            loss = criterion(logits, y_device)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
        studies.append(study_id.numpy())
        losses.append(float(loss.item()))
    probability = np.concatenate(probabilities)
    y = np.concatenate(labels)
    return {
        "probabilities": probability,
        "labels": y,
        "studies": np.concatenate(studies),
        "expected_grade": probability @ np.arange(4, dtype=float),
        "softmax_metrics": classification_metrics(y, probability.argmax(axis=1)),
        "loss": float(np.mean(losses)),
    }


def make_loader(frame, seed, epoch, train, batch_size, workers):
    dataset = PairedStudyFrameDataset(frame, seed=seed, epoch=epoch, train=train)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch if train else seed + 100000)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_studies,
        generator=generator,
        persistent_workers=False,
    )


def save_predictions(path, result, thresholds):
    output = pd.DataFrame(
        {
            "study_id_int": result["studies"].astype(int),
            "label_id": result["labels"].astype(int),
            "expected_grade": result["expected_grade"],
            "pred_softmax": result["probabilities"].argmax(axis=1),
            "pred_expected_grade_val_thresholds": np.digitize(
                result["expected_grade"], thresholds
            ),
        }
    )
    for index, column in enumerate(PROB_COLS):
        output[column] = result["probabilities"][:, index]
    output.sort_values("study_id_int").to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mean"], default="mean")
    parser.add_argument("--study-manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if (args.outdir / "COMPLETED").is_file():
        log("variant already completed")
        return

    seed_everything(args.seed)
    study = pd.read_csv(args.study_manifest)
    required = {"study_id_int", "label_id", "split", "clip_frame_paths"}
    if not required.issubset(study.columns):
        raise ValueError(f"Study manifest missing: {sorted(required - set(study.columns))}")
    if study.study_id_int.duplicated().any():
        raise ValueError("Study manifest has duplicate study IDs")
    if not study.split.isin(["train", "val"]).all():
        raise ValueError("Training manifest must contain only train and val rows")
    train_frame = study[study.split.eq("train")].copy()
    val_frame = study[study.split.eq("val")].copy()
    if not len(train_frame) or not len(val_frame):
        raise ValueError("Train or validation split is empty")
    log(
        f"variant={args.variant} train={len(train_frame)} val={len(val_frame)} "
        f"epochs={args.epochs} batch={args.batch_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalV1b(args.variant, args.seed).to(device)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(4), y=train_frame.label_id
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint_path = args.outdir / "latest_checkpoint.pt"
    best_path = args.outdir / "best_softmax_model.pt"
    history = []
    best_rank = None
    best_epoch = None
    start_epoch = 1
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_rank = checkpoint["best_rank"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = int(checkpoint["epoch"]) + 1
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )
        log(f"resumed epoch={checkpoint['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_loader(
            train_frame,
            args.seed,
            epoch,
            True,
            args.batch_size,
            args.train_workers,
        )
        model.train()
        losses = []
        started = time.time()
        for batch_index, (x, mask, y, _) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(x, mask), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                log(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={np.mean(losses[-100:]):.4f}"
                )
        val_loader = make_loader(
            val_frame,
            args.seed,
            epoch,
            False,
            args.batch_size,
            args.eval_workers,
        )
        validation = evaluate(model, val_loader, device, criterion)
        softmax = validation["softmax_metrics"]
        rank = [softmax["macro_f1"], softmax["qwk"], -softmax["mae_grade"]]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": validation["loss"],
            "val_softmax_macro_f1": softmax["macro_f1"],
            "val_softmax_qwk": softmax["qwk"],
            "val_softmax_mae": softmax["mae_grade"],
            "elapsed_min": float((time.time() - started) / 60),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.outdir / "training_history.csv", index=False)
        if best_rank is None or tuple(rank) > tuple(best_rank):
            best_rank = rank
            best_epoch = epoch
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "variant": args.variant,
                    "seed": args.seed,
                    "epoch": epoch,
                    "softmax_metrics": softmax,
                },
                best_path,
            )
            log(f"saved best softmax epoch={epoch} macro_f1={softmax['macro_f1']:.4f}")
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_rank": best_rank,
                "best_epoch": best_epoch,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
            },
            checkpoint_path,
        )
        log(json.dumps(row))

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_loader = make_loader(
        val_frame,
        args.seed,
        best_epoch,
        False,
        args.batch_size,
        args.eval_workers,
    )
    final = evaluate(model, final_loader, device, criterion)
    thresholds, expected_metrics = choose_thresholds(
        final["labels"], final["expected_grade"]
    )
    payload = {
        "variant": args.variant,
        "checkpoint_selection": "validation softmax macro-F1, then QWK, then MAE",
        "best_epoch": int(best_epoch),
        "softmax_argmax": final["softmax_metrics"],
        "expected_grade_val_thresholds_exploratory": {
            "thresholds": thresholds,
            "metrics": expected_metrics,
        },
        "test_evaluated": False,
        "parameter_counts": {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "temporal": int(sum(parameter.numel() for parameter in model.temporal.parameters())),
        },
    }
    with open(args.outdir / "validation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    save_predictions(args.outdir / "predictions_val.csv", final, thresholds)
    (args.outdir / "COMPLETED").touch()
    log(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
