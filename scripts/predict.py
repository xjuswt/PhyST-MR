#!/usr/bin/env python3
"""Generate study-level probabilities from a trained video expert."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

import train_spatial
import train_spatial_physio
import train_temporal


PROB_COLS = ["prob_none_trace", "prob_mild", "prob_moderate", "prob_severe"]


def checked_frame(path, expert):
    frame = pd.read_csv(path)
    identity = "study_id_int" if expert != "temporal" else "sample_id"
    required = {identity, "clip_frame_paths" if expert != "temporal" else "frame_paths_json"}
    if expert == "temporal":
        required.add("study_id")
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    if frame[identity].duplicated().any():
        raise ValueError(f"Manifest has duplicate {identity}")
    has_labels = "label_id" in frame.columns
    if not has_labels:
        frame["label_id"] = 0  # Dataset compatibility; not emitted as a label.
    elif not frame.label_id.isin(range(4)).all():
        raise ValueError("Labels must be 0, 1, 2, or 3")
    return frame, has_labels


@torch.inference_mode()
def predict_spatial(frame, has_labels, args, device):
    if not has_labels:
        frame = frame.assign(label_id=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = checkpoint.get("variant", "mean")
    seed = int(checkpoint.get("seed", 42))
    model_class = (
        train_spatial_physio.TemporalV1b
        if args.expert == "spatial-physio"
        else train_spatial.TemporalV1b
    )
    model = model_class(variant, seed, imagenet=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = train_spatial.make_loader(
        frame, seed, 0, False, args.batch_size, args.workers
    )
    rows = []
    for video, mask, labels, studies in loader:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            output = model(video.to(device), mask.to(device))
            logits = output[0] if isinstance(output, tuple) else output
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        for index, study_id in enumerate(studies.tolist()):
            row = {"study_id_int": int(study_id)}
            if has_labels:
                row["label_id"] = int(labels[index])
            row.update(zip(PROB_COLS, probabilities[index].tolist()))
            row["pred_softmax"] = int(probabilities[index].argmax())
            rows.append(row)
    return pd.DataFrame(rows).sort_values("study_id_int")


@torch.inference_mode()
def predict_temporal(frame, has_labels, args, device):
    if not has_labels:
        frame = frame.assign(label_id=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = r2plus1d_18(weights=None, num_classes=4).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = train_temporal.ClipDataset(frame, "L3", 42, 0, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    rows = []
    for video, labels, studies, samples in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(video.to(device)).float().cpu().numpy()
        for index, study_id in enumerate(studies.tolist()):
            rows.append([study_id, int(samples[index]), int(labels[index]), *logits[index]])
    clips = pd.DataFrame(rows, columns=["study_id", "sample_id", "label_id",
                                         "logit_0", "logit_1", "logit_2", "logit_3"])
    grouped = clips.groupby("study_id", sort=True)
    if has_labels and not (grouped.label_id.nunique() == 1).all():
        raise ValueError("Labels differ within a study")
    mean_logits = grouped[[f"logit_{i}" for i in range(4)]].mean()
    probabilities = torch.softmax(torch.from_numpy(mean_logits.to_numpy()), dim=1).numpy()
    result = pd.DataFrame({"study_id": mean_logits.index.to_numpy()})
    if has_labels:
        result["label_id"] = grouped.label_id.first().to_numpy(dtype=int)
    for index, column in enumerate(PROB_COLS):
        result[column] = probabilities[:, index]
    result["pred_softmax"] = probabilities.argmax(axis=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=["spatial", "spatial-physio", "temporal"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    frame, has_labels = checked_frame(args.manifest, args.expert)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = (predict_temporal(frame, has_labels, args, device)
              if args.expert == "temporal"
              else predict_spatial(frame, has_labels, args, device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({"expert": args.expert, "studies": len(result),
                      "labels_present": has_labels, "output": str(args.output)}))


if __name__ == "__main__":
    main()
