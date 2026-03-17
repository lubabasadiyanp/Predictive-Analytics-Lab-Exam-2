# Binary Classification Lab Exam

A student submission for a Machine Learning lab exam, implementing a complete binary classification pipeline using Logistic Regression on a two-feature synthetic dataset.

---

## Dataset

**File:** `Lab_Exam_binary_classification_dataset.csv`

| Column | Type | Description |
|---|---|---|
| `Feature1` | float64 | Continuous numeric feature |
| `Feature2` | int64 | Continuous numeric feature |
| `Target` | object | Binary label — `"Yes"` or `"No"` |

- **Raw rows:** 1020 (20 rows have missing `Target` → dropped → **1000 usable rows**)
- **Outlier removal:** 1 extreme `Feature1` value (10000.0) removed via the IQR method → **999 final rows**
- **Class balance:** `No` = 784 (78.5 %), `Yes` = 215 (21.5 %) — moderately imbalanced

---

## Notebook Structure

### 1 — Exploratory Data Analysis

| Step | What it does |
|---|---|
| 1.1 Load data | Reads CSV, prints shape, previews first 10 rows |
| 1.2 Data types & statistics | `.dtypes`, `.describe(include='all')` |
| 1.3 Missing values | Counts NaNs per column; drops rows with missing `Target` |
| 1.4 Outlier detection | IQR fence on `Feature1`; removes 1 extreme row |
| 1.5 Class distribution | Bar chart saved as `class_distribution.png` |
| 1.6 Feature distributions | Histogram + KDE for each feature, saved as `feature_distributions.png` |
| 1.7 Box plots by class | Box plots for each feature split by `Target`, saved as `boxplots_by_class.png` |
| 1.8 Correlation heatmap | Pearson correlation of features and binarised target, saved as `correlation_heatmap.png` |

### 2 — Build a Classification Model

| Step | What it does |
|---|---|
| 2.1 Prepare features | `X = [Feature1, Feature2]`, `y = (Target == "Yes").astype(int)` |
| 2.2 Train / Test split | 80 / 20 stratified split (`random_state=42`) |
| 2.3 Feature scaling | `StandardScaler` fitted on training set only |
| 2.4 Train Logistic Regression | `C=1.0`, `max_iter=1000`, `class_weight="balanced"` |
| 2.5 Cross-validation | 5-fold stratified CV using a `Pipeline` (scaler + classifier) to prevent data leakage |

**CV results (from notebook output):**

```
5-Fold CV Accuracy : 0.5215 ± 0.0325
5-Fold CV ROC-AUC  : 0.5279 ± 0.0267
```

> The low scores reflect that the two features have very weak linear correlation with the target (≈ −0.044 and −0.008). The model is near-random chance for this dataset.

### 3 — Decision Boundary Plot

Visualises the learned logistic regression decision boundary overlaid on the scaled training points.

### 4 — Model Evaluation

- Confusion matrix (`ConfusionMatrixDisplay`)
- Full `classification_report` (precision, recall, F1 per class)
- ROC curve with AUC score

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Steps

1. Place `Lab_Exam_binary_classification_dataset.csv` in the same directory as the notebook.
2. Open the notebook in **Jupyter** or **Google Colab**.
3. Run all cells top-to-bottom (`Runtime → Run all` in Colab, or `Kernel → Restart & Run All` in Jupyter).

All output plots are saved automatically as PNGs in the working directory.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Scaling, splitting, cross-validation, logistic regression, metrics |

---

## Key Design Decisions

- **`class_weight="balanced"`** — automatically adjusts class weights inversely proportional to class frequencies, mitigating the ~78 / 22 class imbalance.
- **Pipeline in CV** — scaler is fitted inside each fold to avoid data leakage from the test fold into the model.
- **Stratified splits** — both the train/test split and the K-Fold use stratification to preserve class proportions.
