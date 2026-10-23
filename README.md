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

> **Note:** the decision threshold is chosen per category and method by maximising F1 on the
> test set. This is a simplification for the prototype and makes the numbers optimistic.

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
        ├── defect_type_metrics.json    (only if the optional section ran)
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
| `threshold` | Threshold used for this category and method |
| `prediction` | `1` if `anomaly_score >= threshold`, else `0` |
| `predicted_defect_type` | Defect type predicted by the optional classifier; falls back to `true_defect_type` when no prediction exists (good images, toothbrush, or optional section removed) |

Tip: filter on one method (e.g. the best one from `summary_table.csv`) to get one row per image.

### `metrics.json`
```json
{
  "hazelnut": {
    "knn_cosine":  {"auc": 0.97, "recall": 0.93, "false_alarm_rate": 0.05, "threshold": 0.12,
                    "f1": 0.94, "tp": 65, "fp": 2, "tn": 38, "fn": 5},
    "mahalanobis": {...},
    "ocsvm":       {...}
  },
  "grid": {...}
}
```
(numbers above are only an illustration). **RL scheduling** mainly needs `recall` (chance that a defect
is caught) and `false_alarm_rate` (chance that a good item is flagged).

### `summary_table.csv` and `plots/`
Summary per category and method, plus `preprocessing_examples.png`, `roc_curves.png`,
`confusion_matrices.png` and `defect_type_confusion.png`.

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
