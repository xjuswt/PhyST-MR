#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASSES = ["None/Trace", "Mild", "Moderate", "Severe"]
PROBS = ["prob_none_trace", "prob_mild", "prob_moderate", "prob_severe"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions(path, prefix):
    frame = pd.read_csv(path).rename(
        columns={"study_id_int": "study_id", "true_label": "label_id"}
    )
    prefixed_probabilities = {
        f"r2d_{column}": column for column in PROBS
    }
    if not set(PROBS).issubset(frame.columns) and set(prefixed_probabilities).issubset(frame.columns):
        frame = frame.rename(columns=prefixed_probabilities)
    missing = {"study_id", "label_id", *PROBS} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.study_id.duplicated().any():
        raise ValueError(f"{path}: duplicate study_id")
    probability = frame[PROBS].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability.sum(axis=1) <= 0).any():
        raise ValueError(f"{path}: invalid probabilities")
    if not frame.label_id.isin(range(4)).all():
        raise ValueError(f"{path}: labels must be 0, 1, 2, or 3")
    probability /= probability.sum(axis=1, keepdims=True)
    result = frame[["study_id", "label_id"]].copy()
    for index in range(4):
        result[f"{prefix}_p{index}"] = probability[:, index]
    return result


def pair(resnet_path, r2d_path, expected_n):
    resnet = read_predictions(resnet_path, "resnet")
    r2d = read_predictions(r2d_path, "r2d")
    if set(resnet.study_id) != set(r2d.study_id):
        raise ValueError(
            "Study sets differ: "
            f"resnet_only={len(set(resnet.study_id)-set(r2d.study_id))}, "
            f"r2d_only={len(set(r2d.study_id)-set(resnet.study_id))}"
        )
    paired = resnet.merge(
        r2d, on="study_id", suffixes=("_resnet", "_r2d"), validate="one_to_one"
    )
    if not (paired.label_id_resnet == paired.label_id_r2d).all():
        raise ValueError("Labels differ between experts")
    paired = paired.rename(columns={"label_id_resnet": "label_id"}).drop(
        columns="label_id_r2d"
    )
    paired = paired.sort_values("study_id").reset_index(drop=True)
    if len(paired) != expected_n:
        raise ValueError(f"Expected {expected_n} studies, got {len(paired)}")
    return paired


def arrays(paired):
    resnet = paired[[f"resnet_p{k}" for k in range(4)]].to_numpy(float)
    r2d = paired[[f"r2d_p{k}" for k in range(4)]].to_numpy(float)
    return paired.label_id.to_numpy(int), resnet, r2d


def fuse(resnet, r2d, weights):
    score = resnet * weights + r2d * (1.0 - weights)
    return score / score.sum(axis=1, keepdims=True)


def metrics(labels, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(4), zero_division=0
    )
    difference = np.abs(labels - prediction)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")),
        "mae": float(difference.mean()),
        "one_off_accuracy": float((difference <= 1).mean()),
        "two_or_more_off": float((difference >= 2).mean()),
        "per_class": {
            CLASSES[k]: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in range(4)
        },
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(4)
        ).tolist(),
    }


def candidate_grid(values):
    return np.asarray(list(itertools.product(*values)), dtype=np.float64)


def batch_search(labels, resnet, r2d, candidates, batch_size=2048):
    n = len(labels)
    support = np.bincount(labels, minlength=4).astype(float)
    grade = np.arange(4)
    squared_cost = np.square(grade[:, None] - grade[None, :])
    absolute_cost = np.abs(grade[:, None] - grade[None, :])
    rows = []
    for start in range(0, len(candidates), batch_size):
        weight = candidates[start : start + batch_size]
        score = (
            resnet[None, :, :] * weight[:, None, :]
            + r2d[None, :, :] * (1.0 - weight[:, None, :])
        )
        prediction = score.argmax(axis=2)
        matrix = np.zeros((len(weight), 4, 4), dtype=np.int32)
        for true_class in range(4):
            mask = labels == true_class
            for predicted_class in range(4):
                matrix[:, true_class, predicted_class] = (
                    prediction[:, mask] == predicted_class
                ).sum(axis=1)
        predicted_support = matrix.sum(axis=1).astype(float)
        diagonal = np.diagonal(matrix, axis1=1, axis2=2).astype(float)
        denominator = support[None, :] + predicted_support
        f1 = np.divide(
            2.0 * diagonal,
            denominator,
            out=np.zeros_like(diagonal),
            where=denominator > 0,
        )
        macro_f1 = f1.mean(axis=1)
        mae = (matrix * absolute_cost[None, :, :]).sum(axis=(1, 2)) / n
        observed = (matrix * squared_cost[None, :, :]).sum(axis=(1, 2))
        expected = np.einsum(
            "i,bj,ij->b", support, predicted_support, squared_cost
        ) / n
        qwk = 1.0 - np.divide(
            observed.astype(float),
            expected,
            out=np.zeros_like(expected, dtype=float),
            where=expected > 0,
        )
        distance = np.square(weight - 0.5).sum(axis=1)
        accuracy = diagonal.sum(axis=1) / n
        for index in range(len(weight)):
            rows.append(
                {
                    "w0": float(weight[index, 0]),
                    "w1": float(weight[index, 1]),
                    "w2": float(weight[index, 2]),
                    "w3": float(weight[index, 3]),
                    "macro_f1": float(macro_f1[index]),
                    "qwk": float(qwk[index]),
                    "mae": float(mae[index]),
                    "accuracy": float(accuracy[index]),
                    "distance_to_equal_weights": float(distance[index]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [
            "macro_f1",
            "qwk",
            "mae",
            "distance_to_equal_weights",
            "w0",
            "w1",
            "w2",
            "w3",
        ],
        ascending=[False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)


def evaluate(paired, weights):
    labels, resnet, r2d = arrays(paired)
    probability = fuse(resnet, r2d, weights)
    return {
        "spatial_expert": metrics(labels, resnet.argmax(axis=1)),
        "r2plus1d_k400_repeat32": metrics(labels, r2d.argmax(axis=1)),
        "classwise_fusion": metrics(labels, probability.argmax(axis=1)),
    }, probability


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-val", required=True, type=Path)
    parser.add_argument("--resnet-test", required=True, type=Path)
    parser.add_argument("--r2d-val", required=True, type=Path)
    parser.add_argument("--r2d-test", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expected-val", type=int, default=757)
    parser.add_argument("--expected-test", type=int, default=802)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    validation = pair(args.resnet_val, args.r2d_val, args.expected_val)
    validation.to_csv(args.outdir / "paired_validation.csv", index=False)

    labels, resnet, r2d = arrays(validation)
    coarse_values = np.round(np.arange(0.10, 0.9001, 0.05), 2)
    coarse = batch_search(
        labels, resnet, r2d, candidate_grid([coarse_values] * 4)
    )
    coarse.to_csv(args.outdir / "weight_search_coarse.csv", index=False)
    center = coarse.loc[0, ["w0", "w1", "w2", "w3"]].to_numpy(float)
    fine_values = [
        np.round(
            np.arange(
                max(0.10, value - 0.10),
                min(0.90, value + 0.10) + 0.0001,
                0.01,
            ),
            2,
        )
        for value in center
    ]
    fine = batch_search(labels, resnet, r2d, candidate_grid(fine_values))
    fine.to_csv(args.outdir / "weight_search_fine.csv", index=False)
    weights = fine.loc[0, ["w0", "w1", "w2", "w3"]].to_numpy(float)

    locked = {
        "class_order": CLASSES,
        "resnet_weights": weights.tolist(),
        "r2d_weights": (1.0 - weights).tolist(),
        "search_split": f"Validation={len(validation)}",
        "coarse_step": 0.05,
        "fine_step": 0.01,
        "bounds": [0.10, 0.90],
        "selection": [
            "Macro-F1 descending",
            "QWK descending",
            "MAE ascending",
            "squared distance to [0.5,0.5,0.5,0.5] ascending",
        ],
        "test_used_for_weight_search": False,
    }
    dump(args.outdir / "LOCKED_CLASSWISE_WEIGHTS.json", locked)

    test = pair(args.resnet_test, args.r2d_test, args.expected_test)
    if set(validation.study_id) & set(test.study_id):
        raise ValueError("Validation and test study sets overlap")
    test.to_csv(args.outdir / "paired_test.csv", index=False)
    validation_metrics, _ = evaluate(validation, weights)
    test_metrics, test_probability = evaluate(test, weights)
    dump(args.outdir / "metrics_validation.json", validation_metrics)
    dump(args.outdir / "metrics_test.json", test_metrics)

    output = test.copy()
    for index in range(4):
        output[f"fusion_p{index}"] = test_probability[:, index]
    output["pred_fusion"] = test_probability.argmax(axis=1)
    output.to_csv(args.outdir / "predictions_test_fusion.csv", index=False)
    pd.DataFrame(
        test_metrics["classwise_fusion"]["confusion_matrix"],
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(args.outdir / "confusion_matrix_test.csv")
    pd.DataFrame(test_metrics["classwise_fusion"]["per_class"]).T.to_csv(
        args.outdir / "per_class_test.csv"
    )

    audit = {
        "validation_n": len(validation),
        "test_n": len(test),
        "study_and_label_alignment": "PASS",
        "sources": {
            str(path): sha256(path)
            for path in [
                args.resnet_val,
                args.resnet_test,
                args.r2d_val,
                args.r2d_test,
            ]
        },
    }
    dump(args.outdir / "INPUT_AUDIT.json", audit)

    def row(name, value):
        return (
            f"| {name} | {value['accuracy']:.4f} | "
            f"{value['balanced_accuracy']:.4f} | {value['macro_f1']:.4f} | "
            f"{value['weighted_f1']:.4f} | {value['qwk']:.4f} | "
            f"{value['mae']:.4f} | {100*value['two_or_more_off']:.2f}% |"
        )

    report = f"""# PhyST-MR class-wise probability fusion

The four class-specific weights were selected only on the locked Validation set
and were fixed before Test evaluation. Test labels were not used for weight
selection.

## Locked weights

- Class order: {CLASSES}
- ResNet18 weights: {weights.tolist()}
- R(2+1)D weights: {(1.0-weights).tolist()}

## Validation

| Model | Acc | BAcc | Macro-F1 | Weighted-F1 | QWK | MAE | >=2 error |
|---|---:|---:|---:|---:|---:|---:|---:|
{row('Physiology-guided spatial expert', validation_metrics['spatial_expert'])}
{row('R(2+1)D-K400 repeat32', validation_metrics['r2plus1d_k400_repeat32'])}
{row('Class-wise fusion', validation_metrics['classwise_fusion'])}

## Test

| Model | Acc | BAcc | Macro-F1 | Weighted-F1 | QWK | MAE | >=2 error |
|---|---:|---:|---:|---:|---:|---:|---:|
{row('Physiology-guided spatial expert', test_metrics['spatial_expert'])}
{row('R(2+1)D-K400 repeat32', test_metrics['r2plus1d_k400_repeat32'])}
{row('Class-wise fusion', test_metrics['classwise_fusion'])}
"""
    (args.outdir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (args.outdir / "COMPLETED").touch()
    print(json.dumps({"locked_weights": locked, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
