"""
src/visualization/ml_figures.py
================================
Generates the classical ML model figures:

  Figures 7-14  - Base vs tuned accuracy bar for each of the 7 models
                  (Figure 10 is the RF feature importance chart, not a bar)
  Figure 10     - Random Forest feature importance ranking
  Figure 15     - All 7 models compared side by side (grouped bar)
  Figure 16     - ROC curves for all 7 classical models

These figures support Section IV-A of the thesis by showing how each
classical model performed and how tuning affected the results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from sklearn.metrics import roc_curve, roc_auc_score

from config import (DARK_BG, NAVY, BLUE, TEAL, RED, AMBER, GREEN,
                    WHITE, GREY, FONT, FEATURES, BAR_COLORS,
                    MODEL_FIG_NUM, MODEL_COLORS)
from src.utils import save_fig, style_ax


def figures_7_to_14_bar_charts(base_acc: dict, tuned_acc: dict) -> None:
    """
    Figures 7, 8, 9, 11, 12, 13, 14 - one bar chart per classical model.

    Each chart shows two bars side by side: the base model accuracy
    on the left and the tuned model accuracy on the right. The accuracy
    delta is shown as a badge at the bottom of the chart.

    Figure 10 is the feature importance chart and is generated
    separately by figure10_feature_importance() below.
    """
    for model_name in BAR_COLORS:
        fig_n      = MODEL_FIG_NUM[model_name]
        c_base, c_tuned = BLUE, TEAL
        ba = base_acc[model_name]
        ta = tuned_acc[model_name]

        fig, ax = plt.subplots(figsize=(7.5, 5.4), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax)

        labels = [f"{model_name}\n(Base Model)",
                  f"{model_name}\n(Tuned Model)"]
        bars   = ax.bar(labels, [ba, ta],
                        color=[c_base, c_tuned],
                        width=0.46,
                        edgecolor=WHITE,
                        linewidth=0.8,
                        zorder=3)

        # Value labels above each bar
        for bar, val in zip(bars, [ba, ta]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.010,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=14, fontweight="bold",
                    color=WHITE, fontfamily=FONT)

        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.10))
        ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.10)],
                           color=WHITE, fontsize=9.5)
        ax.set_xticklabels(labels, color=WHITE, fontsize=11,
                           fontfamily=FONT)
        ax.set_ylabel("Test Accuracy", color=WHITE, fontsize=11,
                      fontfamily=FONT)
        ax.set_title(
            f"Figure {fig_n}.  {model_name}  -  Base vs Tuned Accuracy",
            color=WHITE, fontsize=12, fontweight="bold",
            pad=14, fontfamily=FONT)

        # 0.75 reference line
        ax.axhline(0.75, color=AMBER, lw=1.4, ls="--", alpha=0.60, zorder=2)
        ax.text(1.015, 0.75, "0.75",
                transform=ax.get_yaxis_transform(),
                color=AMBER, fontsize=8.5, va="center")

        # Delta badge
        diff      = ta - ba
        sign      = "+" if diff >= 0 else ""
        badge_col = TEAL if diff >= 0 else RED
        ax.text(0.5, 0.07,
                f"Accuracy change :  {sign}{diff:.4f}",
                transform=ax.transAxes,
                ha="center", fontsize=11,
                color=badge_col, fontfamily=FONT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=NAVY,
                          alpha=0.75, edgecolor=badge_col, lw=1.2))

        fname = (f"Figure{fig_n}_"
                 + model_name.replace(" ", "_")
                 + "_Base_vs_Tuned.png")
        plt.tight_layout(pad=1.8)
        save_fig(fname, fig)


def figure10_feature_importance(rf_model) -> None:
    """
    Figure 10 - Random Forest feature importance ranking.

    Shows the mean decrease in impurity for each of the eight PIMA
    features from the tuned Random Forest model. The top three most
    important features are highlighted with an amber border.
    This confirms whether the model learned clinically meaningful
    patterns (Glucose, BMI, and Age should rank highest).
    """
    importances = rf_model.feature_importances_
    sorted_idx  = np.argsort(importances)
    s_feats     = [FEATURES[i] for i in sorted_idx]
    s_vals      = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    ax.xaxis.grid(True, linestyle="--", alpha=0.20, color=WHITE)
    ax.yaxis.grid(False)

    bar_cols = plt.cm.Blues(np.linspace(0.38, 0.92, len(s_feats)))
    bars     = ax.barh(s_feats, s_vals, color=bar_cols,
                       edgecolor=WHITE, linewidth=0.5,
                       height=0.62, zorder=3)

    for bar, val in zip(bars, s_vals):
        ax.text(val + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=10,
                color=WHITE, fontfamily=FONT)

    # Highlight top 3 with amber border
    for i in range(len(s_feats) - 3, len(s_feats)):
        bars[i].set_edgecolor(AMBER)
        bars[i].set_linewidth(2.2)

    ax.set_xlabel("Mean Decrease in Impurity",
                  color=WHITE, fontsize=11, fontfamily=FONT)
    ax.set_title("Figure 10.  Random Forest  -  Feature Importance Ranking",
                 color=WHITE, fontsize=12, fontweight="bold",
                 pad=14, fontfamily=FONT)
    ax.set_xlim(0, s_vals.max() * 1.25)
    ax.set_yticklabels(s_feats, color=WHITE, fontsize=11, fontfamily=FONT)
    ax.text(0.98, 0.04,
            "Top 3 features marked with amber border",
            transform=ax.transAxes, ha="right",
            fontsize=9, color=AMBER, fontfamily=FONT)

    plt.tight_layout(pad=1.8)
    save_fig("Figure10_RF_Feature_Importance.png", fig)


def figure15_all_models_accuracy(base_acc: dict, tuned_acc: dict) -> None:
    """
    Figure 15 - All seven classical models compared.

    Horizontal grouped bar chart showing base accuracy and tuned
    accuracy for every model on the same axes. This gives an at-a-glance
    view of which models performed best and how much tuning helped each.
    """
    ml_names = list(BAR_COLORS.keys())
    ba_vals  = [base_acc[n]  for n in ml_names]
    ta_vals  = [tuned_acc[n] for n in ml_names]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    ax.xaxis.grid(True, linestyle="--", alpha=0.22, color=WHITE)
    ax.yaxis.grid(False)

    y  = np.arange(len(ml_names))
    h  = 0.36
    b1 = ax.barh(y + h / 2, ba_vals, h, color=BLUE,
                 label="Base Model",  edgecolor=WHITE, lw=0.5, zorder=3)
    b2 = ax.barh(y - h / 2, ta_vals, h, color=TEAL,
                 label="Tuned Model", edgecolor=WHITE, lw=0.5, zorder=3)

    for bar, val in zip(list(b1) + list(b2), ba_vals + ta_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9,
                color=WHITE, fontfamily=FONT)

    ax.set_yticks(y)
    ax.set_yticklabels(ml_names, color=WHITE, fontsize=11, fontfamily=FONT)
    ax.set_xlabel("Test Accuracy", color=WHITE, fontsize=11, fontfamily=FONT)
    ax.set_xlim(0, 1.04)
    ax.axvline(0.75, color=AMBER, lw=1.3, ls="--", alpha=0.55)
    ax.text(0.754, -0.65, "0.75 ref", color=AMBER,
            fontsize=8.5, fontfamily=FONT)
    ax.set_title(
        "Figure 15.  All Classical ML Models  -  Base vs Tuned Accuracy",
        color=WHITE, fontsize=12, fontweight="bold",
        pad=14, fontfamily=FONT)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.35,
              facecolor=NAVY, labelcolor=WHITE)

    plt.tight_layout(pad=1.8)
    save_fig("Figure15_All_ML_Accuracy.png", fig)


def figure16_roc_curves(tuned_models: dict,
                         X_test: np.ndarray,
                         y_test: np.ndarray) -> None:
    """
    Figure 16 - ROC curves for all seven classical models.

    All seven ROC curves are drawn on the same axes so the relative
    discriminative performance of each model can be compared directly.
    The model with the highest AUC is drawn with a thicker line.
    AUC values are shown in the legend.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    ax.xaxis.grid(True, linestyle="--", alpha=0.18, color=WHITE)

    auc_vals = {
        name: roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        for name, model in tuned_models.items()
    }
    max_auc = max(auc_vals.values())

    for (name, model), col in zip(tuned_models.items(), MODEL_COLORS[:7]):
        yprob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        auc         = auc_vals[name]
        lw          = 2.8 if auc == max_auc else 1.9
        ax.plot(fpr, tpr, color=col, lw=lw,
                label=f"{name}   (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1.1, alpha=0.40,
            label="Random classifier  (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=WHITE)

    ax.set_xlabel("False Positive Rate", color=WHITE, fontsize=12,
                  fontfamily=FONT)
    ax.set_ylabel("True Positive Rate",  color=WHITE, fontsize=12,
                  fontfamily=FONT)
    ax.set_title("Figure 16.  ROC Curves  -  All Seven Classical ML Models",
                 color=WHITE, fontsize=12, fontweight="bold",
                 pad=14, fontfamily=FONT)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.35,
              facecolor=NAVY, labelcolor=WHITE)

    plt.tight_layout(pad=1.8)
    save_fig("Figure16_ROC_Curves.png", fig)


def generate_all(base_acc: dict, tuned_acc: dict, tuned_models: dict,
                 rf_model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Generate and save Figures 7 through 16."""
    print("  Figures 7-9, 11-14 - Individual model bar charts")
    figures_7_to_14_bar_charts(base_acc, tuned_acc)
    print("  Figure 10 - Random Forest feature importance")
    figure10_feature_importance(rf_model)
    print("  Figure 15 - All models accuracy comparison")
    figure15_all_models_accuracy(base_acc, tuned_acc)
    print("  Figure 16 - ROC curves")
    figure16_roc_curves(tuned_models, X_test, y_test)
