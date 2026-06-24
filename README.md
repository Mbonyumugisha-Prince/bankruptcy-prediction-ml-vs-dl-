# Bankruptcy Prediction: Machine Learning vs. Deep Learning

A comparative study predicting corporate bankruptcy from financial ratios, benchmarking traditional machine learning (Scikit-learn) against deep learning (TensorFlow) approaches on a severely imbalanced, real-world dataset.

## Problem

Can a company's financial ratios — profitability, leverage, liquidity — predict bankruptcy before it happens? This project builds and compares nine modeling experiments across two paradigms to find out, with a deliberate focus on how severe class imbalance (only 3.23% of companies in the dataset went bankrupt) affects model choice, optimization, and evaluation.

## Dataset

**[Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)** — sourced from the Taiwan Economic Journal, covering 6,819 companies from 1999–2009, with 94 financial ratio features after cleaning.

- 96.77% solvent / 3.23% bankrupt (220 companies)
- No missing values or duplicates; one constant column dropped
- 23 feature pairs with correlation > 0.95 retained (handled via tree-based importance and regularization rather than manual pruning)

## Approaches compared

**Traditional ML (Scikit-learn):**

| Experiment | Model | Notes |
|---|---|---|
| E1 | Logistic Regression | class-balanced |
| E2 | Logistic Regression | unbalanced (control) |
| E3 | Random Forest | 200 trees, balanced |
| E4 | Random Forest | tuned via GridSearchCV |
| E5 | Gradient Boosting | default settings |

**Deep Learning (TensorFlow, tf.data API):**

| Experiment | Architecture | Notes |
|---|---|---|
| E6 | Sequential | 30% dropout |
| E7 | Sequential | 20% dropout |
| E8 | Functional API | + Batch Normalization |
| E9 | Functional API | + Batch Normalization, higher learning rate |

## Results

| Model | ROC-AUC | Recall | False Alarms |
|---|---|---|---|
| **E4 — Random Forest (tuned)** | **0.939** | 0.606 | 37 |
| E5 — Gradient Boosting | 0.938 | 0.333 | — |
| E3 — Random Forest (default) | 0.937 | 0.424 | — |
| **E7 — Sequential (20% dropout)** | **0.930** | **0.849** | 141 |

Full results, ROC curves, confusion matrices, and learning curves for all nine experiments are in the notebook and written report.

**Key finding:** Tree-based ensembles and neural networks reach near-identical ranking ability (ROC-AUC), but diverge sharply in their precision-recall trade-off at a fixed decision threshold — the best deep learning model catches far more true bankruptcies (28/33 vs. 20/33) but at nearly four times the false-alarm rate.

**Recommendation:** This project deploys **E4 (tuned Random Forest)** — best ROC-AUC, a manageable false-alarm rate, and feature importances that are easier to justify to a regulator than a neural network's predictions.

## Repository contents

```
├── NotBook.ipynb         # Full modular, documented pipeline: preprocessing,
│                          # 9 experiments, evaluation, error analysis
├── Final_Report.docx     # Written report (IEEE citations, ~5,000 words)
├── figures/               # ROC curve, confusion matrices, learning curves,
│                          # feature importance chart
├── requirements.txt
└── README.md
```

## Running the notebook

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
jupyter notebook NotBook.ipynb
```

Run top-to-bottom; random seeds are set for reproducibility.

## Links

- **Report:** `Final_Report.docx` (this repo)
- **Presentation video:** [insert link]


---
*Final summative project — Introduction to Machine Learning*

