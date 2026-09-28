import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.video import r2plus1d_18

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import apply_fusion
import classwise_fusion
import predict
import train_spatial
import train_spatial_physio
import train_temporal


class PipelineTests(unittest.TestCase):
    def test_full_finetuning_and_checkpoint_compatibility(self):
        spatial = train_spatial.TemporalV1b("mean", 42, imagenet=False)
        self.assertTrue(all(p.requires_grad for p in spatial.encoder.parameters()))
        physiology = train_spatial_physio.TemporalV1b("mean", 42, imagenet=False)
        mismatch = physiology.load_state_dict(spatial.state_dict(), strict=False)
        self.assertEqual(set(mismatch.missing_keys), {
            "physiology_head.0.weight", "physiology_head.0.bias",
            "physiology_head.2.weight", "physiology_head.2.bias",
        })
        self.assertEqual(mismatch.unexpected_keys, [])

    def test_temporal_repeat_16_to_32(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(16):
                path = Path(directory) / f"{index:02d}.png"
                Image.new("RGB", (8, 8), (index * 8, 0, 0)).save(path)
                paths.append(str(path))
            frame = pd.DataFrame([{
                "study_id": 1, "sample_id": 2, "label_id": 3,
                "frame_paths_json": json.dumps(paths),
            }])
            video, label, study, sample = train_temporal.ClipDataset(
                frame, "L3", 42, 0, False
            )[0]
            self.assertEqual(tuple(video.shape), (3, 32, 8, 8))
            self.assertTrue(torch.equal(video[:, 0], video[:, 1]))
            self.assertTrue(torch.equal(video[:, 30], video[:, 31]))
            self.assertEqual((label, study, sample), (3, 1, 2))

    def test_spatial_physio_inference_needs_no_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            frame = pd.DataFrame([{
                "study_id_int": 7,
                "clip_frame_paths": json.dumps([[str(image)] * 16]),
                "label_id": 0,
            }])
            model = train_spatial_physio.TemporalV1b("mean", 42, imagenet=False)
            checkpoint = root / "model.pt"
            torch.save({"model": model.state_dict(), "variant": "mean", "seed": 42}, checkpoint)
            output = predict.predict_spatial(
                frame.drop(columns="label_id"), False,
                SimpleNamespace(expert="spatial-physio", checkpoint=checkpoint,
                                batch_size=1, workers=0),
                torch.device("cpu"),
            )
            self.assertEqual(output.study_id_int.tolist(), [7])
            self.assertNotIn("label_id", output)
            self.assertAlmostEqual(float(output[classwise_fusion.PROBS].sum(axis=1).iloc[0]), 1.0, places=6)

    def test_temporal_inference_study_level_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            frame = pd.DataFrame([{
                "study_id": 7, "sample_id": 11,
                "frame_paths_json": json.dumps([str(image)] * 16),
            }])
            model = r2plus1d_18(weights=None, num_classes=4)
            checkpoint = root / "model.pt"
            torch.save({"model": model.state_dict()}, checkpoint)
            del model
            output = predict.predict_temporal(
                frame, False,
                SimpleNamespace(checkpoint=checkpoint, batch_size=1, workers=0),
                torch.device("cpu"),
            )
            self.assertEqual(output.study_id.tolist(), [7])
            self.assertNotIn("label_id", output)
            self.assertAlmostEqual(float(output[classwise_fusion.PROBS].sum(axis=1).iloc[0]), 1.0, places=6)

    def test_fusion_formula_and_unlabeled_application(self):
        spatial = np.array([[0.8, 0.1, 0.05, 0.05]])
        temporal = np.array([[0.2, 0.1, 0.3, 0.4]])
        weights = np.array([0.25, 0.5, 0.75, 0.5])
        expected_q = weights * spatial + (1 - weights) * temporal
        np.testing.assert_allclose(
            classwise_fusion.fuse(spatial, temporal, weights),
            expected_q / expected_q.sum(axis=1, keepdims=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = dict(zip(classwise_fusion.PROBS, spatial[0]))
            pd.DataFrame([{"study_id_int": 99, **columns}]).to_csv(root / "s.csv", index=False)
            columns = dict(zip(classwise_fusion.PROBS, temporal[0]))
            pd.DataFrame([{"study_id": 99, **columns}]).to_csv(root / "t.csv", index=False)
            (root / "w.json").write_text(json.dumps({"resnet_weights": weights.tolist()}))
            argv = ["apply_fusion.py", "--spatial", str(root / "s.csv"),
                    "--temporal", str(root / "t.csv"), "--weights-json",
                    str(root / "w.json"), "--output", str(root / "out.csv")]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                apply_fusion.main()
            result = pd.read_csv(root / "out.csv")
            self.assertEqual(result.study_id.tolist(), [99])
            self.assertNotIn("label_id", result)
            np.testing.assert_allclose(result[classwise_fusion.PROBS].to_numpy(),
                                       expected_q / expected_q.sum())

    def test_fusion_search_uses_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.tile(np.arange(4), 2)
            for split, offset in (("val", 100), ("test", 200)):
                for expert in ("s", "t"):
                    rows = []
                    for index, label in enumerate(labels):
                        probability = np.full(4, 0.1)
                        predicted = label if expert == "s" or index % 3 else (label + 1) % 4
                        probability[predicted] = 0.7
                        rows.append({"study_id": offset + index, "label_id": int(label),
                                     **dict(zip(classwise_fusion.PROBS, probability))})
                    pd.DataFrame(rows).to_csv(root / f"{expert}_{split}.csv", index=False)
            argv = ["classwise_fusion.py", "--resnet-val", str(root / "s_val.csv"),
                    "--r2d-val", str(root / "t_val.csv"), "--resnet-test",
                    str(root / "s_test.csv"), "--r2d-test", str(root / "t_test.csv"),
                    "--expected-val", "8", "--expected-test", "8",
                    "--outdir", str(root / "fusion")]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                classwise_fusion.main()
            locked = json.loads((root / "fusion" / "LOCKED_CLASSWISE_WEIGHTS.json").read_text())
            self.assertEqual(len(locked["resnet_weights"]), 4)
            self.assertEqual(locked["search_split"], "Validation=8")
            self.assertEqual(len(pd.read_csv(root / "fusion" / "predictions_test_fusion.csv")), 8)


if __name__ == "__main__":
    unittest.main()
