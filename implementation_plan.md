# New Paper Figures — SignBridge ASL Model Evaluation

## Background

You have real eval outputs in `asl-signs/eval_outputs/`:
- `classification_report.txt` — per-class Precision, Recall, F1, Support for **250 classes**
- `confusion_matrix_full.csv` — 250×250 confusion matrix
- `evaluation_summary.json` — Top-1 (71.38%), Top-5 (89.61%), Macro P/R/F1, Weighted P/R/F1

You already have **9 figures** in `paper_figures/`. The new figures are drawn from the classification report's **per-class P/R/F1** data — the type of visualizations common in NLP/vision papers.

---

## Proposed New Figures (from classification_report.txt)

All figures match the style of the existing `paper_figures/` (300 dpi PNG, DejaVu Sans, clean spines).

---

### Fig 10 — Precision vs Recall Scatter (all 250 classes)

**Type**: Scatter plot coloured by F1-score  
**What it shows**: Each sign is a dot at (Precision, Recall). Colour = F1. A diagonal line shows P=R. Signs far off-diagonal have P↔R imbalance. Signs in bottom-left corner are "hard" signs.  
**Why useful**: Classic diagnostic figure in ML papers — shows the P-R trade-off across all classes at once.

---

### Fig 11 — F1-Score Distribution Histogram + KDE

**Type**: Histogram with overlaid density curve (KDE)  
**What it shows**: Distribution of per-class F1 scores. Vertical lines at mean (70.88%) and ±1 std. Annotates % of classes above/below a threshold (e.g. F1 < 0.5 = "poor").  
**Why useful**: Reveals the spread and skew of model performance — a single macro-F1 hides this.

---

### Fig 12 — Per-Class Precision / Recall / F1 Grouped Bar (Top & Bottom 15)

**Type**: Grouped horizontal bar chart (3 bars per sign: P, R, F1)  
**What it shows**: Side-by-side P, R, F1 for the **15 best** and **15 worst** signs (by F1). Shows whether P or R is the bottleneck per sign.  
**Why useful**: More informative than accuracy-only bars. Directly answers "is recall or precision the problem?"

---

### Fig 13 — Precision–Recall Gap Waterfall / Sorted Bar

**Type**: Sorted bar chart where bar height = |Precision − Recall| per class  
**What it shows**: Signs sorted by the magnitude of P-R gap. Colour encodes direction: blue = precision > recall (over-predictor), red = recall > precision (under-predictor).  
**Why useful**: Identifies systematic biases — some signs are chronically over-predicted or missed.

---

### Fig 14 — F1 Score Heatmap (semantic categories)

**Type**: Square heatmap of F1 scores laid out in a 16×16 grid of all 250 signs  
**What it shows**: Each cell = one sign class, colour = F1. Sorted by F1 so gradients are visible. Acts as a visual "report card" of the full 250-class model.  
**Why useful**: Gives an at-a-glance overview of the entire vocabulary performance — a common figure in large-vocabulary recognition papers.

---

## File Layout

| Figure | Filename | Source data |
|--------|----------|-------------|
| Fig 10 | `fig10_precision_recall_scatter.png` | `classification_report.txt` |
| Fig 11 | `fig11_f1_distribution.png` | `classification_report.txt` |
| Fig 12 | `fig12_prf1_grouped_bars.png` | `classification_report.txt` |
| Fig 13 | `fig13_pr_gap_sorted.png` | `classification_report.txt` |
| Fig 14 | `fig14_f1_heatmap_grid.png` | `classification_report.txt` |

A new script `signbridge_new_figures.py` will be created in `asl-signs/`.

---

## Open Questions

> [!IMPORTANT]
> **Do you want all 5 new figures, or a subset?**
> - Fig 10 (Precision–Recall scatter) — most common in papers
> - Fig 11 (F1 distribution histogram) — good for analysis sections
> - Fig 12 (Grouped P/R/F1 bars for top/bottom 15) — very paper-friendly
> - Fig 13 (P-R gap waterfall) — more niche but reveals bias
> - Fig 14 (F1 heatmap grid of all 250 signs) — good as a supplementary figure

> [!NOTE]
> All figures will use 300 dpi and match the visual style of the existing 9 figures (same fonts, color palette, clean spines). Let me know if you want a different style.
