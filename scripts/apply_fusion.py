#!/usr/bin/env python3
"""Apply locked class-wise probability weights without fitting or requiring labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classwise_fusion import PROBS, fuse


def load_probabilities(path, prefix):
    frame = pd.read_csv(path).rename(columns={"study_id_int": "study_id"})
    required = {"study_id", *PROBS}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(frame.columns))}")
    if frame.study_id.duplicated().any():
        raise ValueError(f"{path}: duplicate study IDs")
    probability = frame[PROBS].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability.sum(axis=1) <= 0).any():
        raise ValueError(f"{path}: invalid probabilities")
    probability /= probability.sum(axis=1, keepdims=True)
    result = frame[["study_id"] + (["label_id"] if "label_id" in frame else [])].copy()
    for index in range(4):
        result[f"{prefix}_p{index}"] = probability[:, index]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locked = json.loads(args.weights_json.read_text(encoding="utf-8"))
    weights = np.asarray(locked["resnet_weights"], dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("Expected four spatial weights between 0 and 1")
    spatial = load_probabilities(args.spatial, "spatial")
    temporal = load_probabilities(args.temporal, "temporal")
    if set(spatial.study_id) != set(temporal.study_id):
        raise ValueError("Expert study sets differ")
    paired = spatial.merge(temporal, on="study_id", how="inner", validate="one_to_one",
                           suffixes=("_spatial", "_temporal")).sort_values("study_id")
    if {"label_id_spatial", "label_id_temporal"}.issubset(paired.columns):
        if not (paired.label_id_spatial == paired.label_id_temporal).all():
            raise ValueError("Expert labels differ")
        paired = paired.rename(columns={"label_id_spatial": "label_id"}).drop(columns="label_id_temporal")
    elif "label_id_spatial" in paired:
        paired = paired.rename(columns={"label_id_spatial": "label_id"})
    elif "label_id_temporal" in paired:
        paired = paired.rename(columns={"label_id_temporal": "label_id"})
    spatial_prob = paired[[f"spatial_p{i}" for i in range(4)]].to_numpy(float)
    temporal_prob = paired[[f"temporal_p{i}" for i in range(4)]].to_numpy(float)
    fused = fuse(spatial_prob, temporal_prob, weights)
    result = paired[["study_id"] + (["label_id"] if "label_id" in paired else [])].copy()
    for index, column in enumerate(PROBS):
        result[column] = fused[:, index]
    result["pred_softmax"] = fused.argmax(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({"studies": len(result), "output": str(args.output)}))


if __name__ == "__main__":
    main()
