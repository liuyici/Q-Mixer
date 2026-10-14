# Data preparation

## MI feature files

Each entry point reads one precomputed MAT file per held-out subject. Pass the
directory containing these files through `--file_path`. Subject IDs are
one-based and consecutive.

| Entry point | Expected file name | Windows | Features |
| --- | --- | ---: | ---: |
| `BCI_14_7513_sota` | `NEW14_LOSO_target_{subject:02d}_W5_logmap_gfk_k25.mat` | 5 | 120 |
| `BCI_2a9_7922_sota` | `BNCI2014012_LOSO_target_{subject:02d}_W5_logmap_gfk_k10.mat` | 5 | 253 |
| `BCI_IVNO7_8293_sota` | `MI1_LOSO_target_{subject:02d}_W1_logmap_gfk_k10.mat` | 1 | 1770 |
| `BCI_IVA5_8479_sota` | `MI5_LOSO_target_{subject:02d}_W1_logmap_gfk_k25.mat` | 1 | 7021 |

`BNCI2014012` is the existing file prefix used by the BNCI2014001 entry point.
The loaders use the literal `k10`/`k25` suffixes above; the `--k` argument does
not change them.

MATLAB v7 and v7.3 files are supported. Required fields:

| Field | Shape after loading | Description |
| --- | --- | --- |
| `X_split` or `X_all` | `(N, W, F)` | Feature array |
| `Y_split` or `Y_all` | `(N,)` | Class labels |
| `subj_split` or `subject_id` | `(N,)` | Subject IDs |
| `is_target` | `(N,)` | Stored target mask |

Optional metadata: `feat_dim`, `feat_dim_out`, `k`, and `USE_GFK`.
Array axes are inferred from the window count, label count, and feature-width
metadata. The training split is selected by `subject_id` and `args.target`;
the stored `is_target` field is loaded but does not determine the split.

## Normalization and splits

Each source subject and the target subject are standardized separately over
their trials. The existing transformation is `(mean - X) / (std + eps)`.
Target statistics therefore use the full unlabeled target feature array.
Target labels are excluded from optimization batches and are used for
evaluation after each training stage.

The entry points expect the fixed window counts and feature widths in the
table. Changing `--window_size` alone does not resize the classifier input.

## Preprocessing included in the repository

`centroid_align.py` and `logmap.py` provide covariance alignment and tangent-space
mapping utilities. The repository does not include an end-to-end exporter for
the fold-specific `logmap_gfk` files above. File names refer to GFK, but the
exporter and its settings are not included. GFK is not described in the
manuscript's MI preprocessing section.

The MATLAB files under `preprocessing/` are separate raw-signal examples:

- `BCIIV2a.m`: 22 channels at 250 Hz; 1,000 samples from event position +500.
- `BCIIV2b.m`: three channels at 250 Hz; 1,000 samples from event position +750.
- Both call `cheby2(6, 60, [4, 40] / 125)`, apply `filtfilt`, and replace NaNs
  with zero.

These examples do not implement the complete MOABB/notch-filter pipeline
described in the manuscript. The feature-generation settings and fold-specific
alignment must be supplied with the prepared MI data.

The SEED helper functions and preprocessing examples are retained, but none of
the four `main.py` files is a SEED, SEED-IV, or SEED-V training entry point.
JSED data and model code are not part of this release.
