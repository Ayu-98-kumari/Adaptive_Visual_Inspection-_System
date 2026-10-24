# Adaptive Visual Inspection System with RL-Based Inspection Scheduling

M.Sc. team project – a simple working prototype.

```
vision  ──►  RL scheduling  ──►  NLP / RAG reports  ──►  BI dashboard
(step 1)       (step 2)              (step 3)              (step 4)
```

This repository currently contains **step 1: vision** – detecting defective products in images
from the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.
The other steps will be added by teammates and build on the vision outputs described below.

## Step 1 – Vision in a nutshell

| Stage | What happens |
|---|---|
| Preprocessing | Resize to 224×224, ImageNet normalisation, optional histogram equalisation or Gaussian blur |
| Features | Pretrained ResNet-18 (no training), last layer removed → 512-d embedding per image |
| Anomaly detection | kNN (cosine, k=5), Mahalanobis distance, One-Class SVM – trained on good images only |
| Evaluation | AUC, confusion matrix, recall, false-alarm rate per category and method |
| Optional | Logistic regression predicts the defect type of defective images |

Categories used: `grid`, `zipper`, `hazelnut`, `leather`, `toothbrush`.

> **Thresholds:** two per category and method.
> - **Default:** the 95th percentile of the cross-validated good training scores (normalised score ≈ 0.95).
>   It uses no test labels, and about 5% of good parts are expected to be flagged.
> - **Comparison:** the threshold with the best F1 on the test set. It uses test labels, so its numbers are optimistic.
>
> Recall and false-alarm rate are reported for both thresholds.

## How to open the notebook in Colab

The notebook is designed for **Google Colab only** (works the same for Windows and macOS users).

1. Go to <https://colab.research.google.com> → **File → Open notebook → GitHub** tab.
2. Tick **Include private repos** and authorise Colab to access GitHub (first time only).
3. Search for `Ayu-98-kumari/Adaptive_Visual_Inspection-_System`, pick the branch, and open
   `notebooks/step1_vision.ipynb`.
   Or use the direct link (after GitHub authorisation):
   <https://colab.research.google.com/github/Ayu-98-kumari/Adaptive_Visual_Inspection-_System/blob/main/notebooks/step1_vision.ipynb>
4. *(Recommended)* **Runtime → Change runtime type → T4 GPU**. CPU also works, just slower.
5. **Runtime → Run all** and allow Google Drive access when asked.

To save your changes back to GitHub use **File → Save a copy in GitHub** (clear outputs first:
**Edit → Clear all outputs**).

All paths and settings are in the **config cell** at the top of the notebook.

## Google Drive folder layout

The dataset and all outputs live in Google Drive, never in this repository.

```
MyDrive/Integrated/
├── data/
│   └── mvtec/
│       ├── grid/
│       │   ├── train/good/
│       │   ├── test/good/
│       │   ├── test/<defect_type>/     e.g. bent, broken, ...
│       │   └── ground_truth/           (masks – not used)
│       ├── zipper/  hazelnut/  leather/  toothbrush/   (same layout)
└── outputs/
    └── vision/                         ← created by the notebook
        ├── anomaly_scores.csv
        ├── metrics.json
        ├── summary_table.csv
        ├── method_choice.json
        ├── defect_type_metrics.json    (only if the optional section ran)
        ├── defect_type_summary.csv     (only if the optional section ran)
        ├── features/                   cached ResNet-18 embeddings (.npy)
        └── plots/                      PNG figures
```

## What the vision step outputs (for teammates)

### `anomaly_scores.csv`
One row per **test image and method** (3 methods → each image appears 3 times).

| Column | Meaning |
|---|---|
| `category` | Product category, e.g. `hazelnut` |
| `image_path` | Path relative to `MyDrive/Integrated/data/mvtec/`, e.g. `hazelnut/test/crack/003.png` |
| `true_defect_type` | Folder name: `good` or the defect type |
| `label` | Ground truth: `0` = good, `1` = defective |
| `method` | `knn_cosine`, `mahalanobis` or `ocsvm` |
| `anomaly_score` | Higher = more likely defective (scales differ between methods) |
| `threshold` | **Default** threshold for this category and method: 95th percentile of good training scores |
| `prediction` | `1` if `anomaly_score >= threshold`, else `0` (**use this one**) |
| `predicted_defect_type` | Defect type predicted by the optional classifier; falls back to `true_defect_type` when no prediction exists (good images, toothbrush, or optional section removed) |
| `normalized_score` | Same score on a common **0–1 scale for all methods**: share of good training images that look less unusual (0.97 = more unusual than 97% of good parts; 1.0 = more unusual than all of them) |
| `normalized_threshold` | The default threshold on that 0–1 scale (≈ 0.95) |
| `threshold_f1thr` | Comparison threshold: best F1 on the test set (optimistic) |
| `prediction_f1thr` | `1` if `anomaly_score >= threshold_f1thr`, else `0` |
| `normalized_threshold_f1thr` | The best-F1 threshold on the 0–1 scale |

`prediction` is always based on the raw `anomaly_score` and `threshold`; the normalised columns are for
comparing methods and for display. Columns ending in `_f1thr` are for comparison only.

Tip: filter on one method (e.g. the best one from `summary_table.csv`) to get one row per image.

### `metrics.json`
```json
{
  "hazelnut": {
    "knn_cosine":  {"auc": 0.97, "recall": 0.93, "false_alarm_rate": 0.05, "f1": 0.94,
                    "tp": 65, "fp": 2, "tn": 38, "fn": 5,
                    "threshold": 0.12, "normalized_threshold": 0.95,
                    "recall_f1thr": 0.95, "false_alarm_rate_f1thr": 0.03, "f1_f1thr": 0.96,
                    "tp_f1thr": 67, "fp_f1thr": 1, "tn_f1thr": 39, "fn_f1thr": 3,
                    "threshold_f1thr": 0.13, "normalized_threshold_f1thr": 0.98},
    "mahalanobis": {...},
    "ocsvm":       {...}
  },
  "grid": {...}
}
```
(numbers above are only an illustration). **RL scheduling** should use `recall` (chance that a defect
is caught) and `false_alarm_rate` (chance that a good item is flagged). These keys, without a suffix, are at the
default 95th-percentile threshold. Keys ending in `_f1thr` are the same numbers at the best-F1 threshold, for comparison.

### `method_choice.json`
Which method we continue with and why: `chosen_method` (highest mean AUC), the default threshold rule, the best
method per category, and mean AUC / recall / false-alarm rate per method at both thresholds.

### `defect_type_metrics.json` / `defect_type_summary.csv` (optional section)
Per category:
- `accuracy`: the average over both halves (train on A and test on B, then the reverse)
- the accuracy of each half (`accuracy_train_a_test_b`, `accuracy_train_b_test_a`)
- `random_guess_accuracy`: what random guessing would give (1 ÷ number of defect types)
- the number of defect types and the size of each half

Toothbrush is skipped.

### `summary_table.csv` and `plots/`
Summary per category and method, plus `preprocessing_examples.png`, `roc_curves.png`,
`confusion_matrices.png`, `score_distributions.png` (all methods on the common 0–1 scale) and
`defect_type_confusion.png`.

## Repository layout

```
notebooks/step1_vision.ipynb   vision notebook (committed with outputs cleared)
requirements.txt               extras beyond Colab defaults (currently none)
.gitignore                     keeps data, outputs, embeddings and checkpoints out of git
```

Rules: never commit data, outputs or `.npy` files; commit notebooks with outputs cleared.

## Local logic check (optional)

Outside Colab, the config switches to `../local_test/` (relative to `notebooks/`, git-ignored).
Put a tiny dummy dataset with the MVTec layout in `local_test/data/mvtec/` to check that the
notebook runs end to end. Results on dummy data are meaningless.
