# Input format

All examples below use fictitious identifiers and paths. File paths inside JSON columns may be absolute or relative to the current working directory. RGB frames are expected to be pre-extracted from the locked clip selection. Do not resample clips differently for the two experts.

## Study manifest (spatial expert)

One row per study. Training requires `study_id_int,label_id,split,clip_frame_paths`. Inference may omit `label_id` and `split`. `clip_frame_paths` is a JSON list of up to ten clips, each represented by a nonempty list of frame paths. The study loader deterministically chooses 16 evenly spaced frames if a clip contains more, or repeats its final frame if fewer are supplied. For the locked protocol, supply exactly 16 frames per clip.

```csv
study_id_int,label_id,split,clip_frame_paths
10001,2,train,"[[""frames/clip_a/00.jpg"",""frames/clip_a/01.jpg"",...],[""frames/clip_b/00.jpg"",...]]"
```

The `...` above is illustrative only and is not valid JSON. Programmatically create the column with `json.dumps(list_of_clip_frame_lists)` and let a CSV writer escape it.

## Clip manifest (temporal expert)

One row per clip. Training requires `study_id,sample_id,label_id,split,frame_paths_json`. Inference may omit `label_id` and `split`. `sample_id` must be unique; every clip of a study must have the same label. `frame_paths_json` must contain **exactly 16** frame paths in temporal order. The temporal loader repeats every frame once, producing 32 steps without interpolation.

```text
study_id=10001
sample_id=50001
label_id=2
split=train
frame_paths_json=["frames/clip_a/00.jpg", ..., "frames/clip_a/15.jpg"]
```

## Structured measurements

One row per **Train** study. Stage 2 requires `study_id` and the ten feature columns below. Missing individual values are allowed and are masked in the loss. Feature means and standard deviations are fitted on **Train only**. Validation measurement rows are optional and used only for auxiliary diagnostic metrics when available; MR checkpoint selection uses video classification metrics. No structured data is needed at inference.

```text
feat_lvedd       feat_lvesd       feat_lvef       feat_la_dimen
feat_la_vol      feat_rv_diam     feat_mv_peak_e  feat_mv_peak_a
feat_mv_peak_e_a feat_tr_mmhg
```

## Prediction CSVs

The prediction scripts emit one row per study, keyed by `study_id_int` (spatial) or `study_id` (temporal), with `prob_none_trace`, `prob_mild`, `prob_moderate`, and `prob_severe`. `label_id` is emitted only if supplied in the manifest. For fusion weight search, both experts must provide labels on the same 757 Validation and 802 Test studies (or pass `--expected-val`/`--expected-test` for a different cohort). `classwise_fusion.py` rejects missing, duplicate, mismatched, or overlapping study IDs. `apply_fusion.py` can run without labels.
