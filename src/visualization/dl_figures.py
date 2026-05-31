"""
src/visualization/dl_figures.py
================================
Generates the deep learning and clinical summary figures:

  Figures 17-19 - Training curves for ANN, DNN, and LSTM individually
  Figure 20     - All three DL models compared side by side
  Figure 21     - Clinical priority ranking by false-negative rate
  Figure 22     - Comprehensive 3-metric comparison (all 10 models)

These figures support Section IV-B and the conclusions of the thesis.
Figure 21 is the most clinically important: it re-ranks all ten models
by false-negative rate to show which models are actually most useful
for diabetes screening, which is different from which models score
highest on accuracy or AUC.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from config import (DARK_BG, NAVY, BLUE, TEAL, RED, AMBER, GREEN,
                    PURPLE, WHITE, GREY, FONT)
from src.utils import save_fig, style_ax


def _training_curves(fig_num: int, wrapper, note: str = "") -> None:
    """
    Draw training and validation curves for one deep learning model.

    Two panels side by side:
      Left  - accuracy over epochs (training and validation)
      Right - loss over epochs (training and validation)

    A vertical dashed line marks the epoch where EarlyStopping fired.
    An optional note string is printed as a small italic caption below
    the figure, used to add context for the LSTM instability.
    """
    hist = wrapper.history.history
    ep   = range(1, len(hist["loss"]) + 1)
    ep_stopped = len(hist["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f"Figure {fig_num}.  {wrapper.name}  -  "
        "Training and Validation  Accuracy and Loss",
        color=WHITE, fontsize=12.5, fontweight="bold",
        fontfamily=FONT, y=1.02)

    panels = [
        ("accuracy",    "val_accuracy",
         "Accuracy", BLUE, TEAL, "Accuracy over Epochs"),
        ("loss",        "val_loss",
         "Loss  (Binary Cross-Entropy)", RED, AMBER, "Loss over Epochs"),
    ]

    for ax, (tr_k, va_k, ylabel, c_tr, c_va, title) in zip(axes, panels):
        style_ax(ax)
        ax.plot(ep, hist[tr_k],  color=c_tr, lw=2.2,
                label=f"Training {ylabel.split()[0]}")
        ax.plot(ep, hist[va_k],  color=c_va, lw=2.2, ls="--",
                label=f"Validation {ylabel.split()[0]}")
        ax.set_xlabel("Epoch",  color=WHITE, fontsize=10.5, fontfamily=FONT)
        ax.set_ylabel(ylabel,   color=WHITE, fontsize=10.5, fontfamily=FONT)
        ax.set_title(title,     color=WHITE, fontsize=11.5, fontfamily=FONT)
        ax.legend(fontsize=9.5, framealpha=0.35,
                  facecolor=NAVY, labelcolor=WHITE)

        # EarlyStopping marker
        ax.axvline(ep_stopped, color=GREY, lw=1.2, ls=":", alpha=0.80)
        y_lo  = ax.get_ylim()[0]
        y_rng = ax.get_ylim()[1] - y_lo
        ax.text(ep_stopped * 0.97,
                y_lo + y_rng * 0.04,
                f"ES @ ep {ep_stopped}",
                color=GREY, fontsize=8, ha="right", fontfamily=FONT)

    if note:
        fig.text(0.5, -0.04, note,
                 ha="center", fontsize=9.5, color=GREY,
                 style="italic", fontfamily=FONT)

    plt.tight_layout(pad=1.8)
    save_fig(f"Figure{fig_num}_{wrapper.name}_Training_Curves.png", fig)


def figure17_ann_curves(ann) -> None:
    """Figure 17 - ANN training and validation accuracy and loss curves."""
    _training_curves(
        17, ann,
        "ANN converges cleanly. Validation accuracy tracks training accuracy "
        "closely, which confirms good generalisation for a model of this size.")


def figure18_dnn_curves(dnn) -> None:
    """Figure 18 - DNN training and validation accuracy and loss curves."""
    _training_curves(
        18, dnn,
        "DNN matches ANN in final accuracy despite added depth and "
        "BatchNorm - direct evidence of diminishing returns from depth "
        "on small static tabular data.")


def figure19_lstm_curves(lstm) -> None:
    """Figure 19 - LSTM training and validation accuracy and loss curves."""
    _training_curves(
        19, lstm,
        "LSTM shows more instability than ANN or DNN. The recurrent gating "
        "mechanism responds to noise in the artificial time-step structure "
        "because the eight PIMA features have no genuine temporal order.")


def figure20_dl_comparison(ann, dnn, lstm) -> None:
    """
    Figure 20 - All three DL models compared across epochs.

    Two panels showing training accuracy (left) and validation accuracy
    (right) for ANN, DNN, and LSTM together. This makes it easy to see
    that ANN is the most stable, DNN is comparable, and LSTM is the
    most erratic throughout training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Figure 20.  All Three DL Models  -  "
        "Training vs Validation Accuracy",
        color=WHITE, fontsize=12.5, fontweight="bold",
        fontfamily=FONT, y=1.02)

    dl_entries = [
        (ann,  BLUE,   "-",  "ANN   (64-32-16)"),
        (dnn,  TEAL,   "--", "DNN   (128-64-32-16)"),
        (lstm, RED,    ":",  "LSTM  (64+32 recurrent layers)"),
    ]

    for side, (metric_key, side_title) in enumerate(
        [("accuracy",     "Training Accuracy"),
         ("val_accuracy", "Validation Accuracy")]
    ):
        ax = axes[side]
        style_ax(ax)
        for wrap, col, ls, label in dl_entries:
            h  = wrap.history.history
            ep = range(1, len(h[metric_key]) + 1)
            ax.plot(ep, h[metric_key],
                    color=col, lw=2.2, ls=ls, label=label)
        ax.set_xlabel("Epoch", color=WHITE, fontsize=10.5, fontfamily=FONT)
        ax.set_ylabel("Accuracy", color=WHITE, fontsize=10.5, fontfamily=FONT)
        ax.set_title(side_title, color=WHITE, fontsize=11.5, fontfamily=FONT)
        ax.legend(fontsize=9.5, framealpha=0.35,
                  facecolor=NAVY, labelcolor=WHITE)
        ax.set_ylim(0.42, 1.00)

    fig.text(
        0.5, -0.04,
        "ANN achieves the highest AUC among DL models.  "
        "LSTM shows the most instability, which is expected because "
        "static PIMA features have no genuine temporal sequence.",
        ha="center", fontsize=9.5, color=GREY,
        style="italic", fontfamily=FONT)

    plt.tight_layout(pad=1.8)
    save_fig("Figure20_DL_Comparison.png", fig)


def figure21_clinical_ranking(results_df: pd.DataFrame) -> None:
    """
    Figure 21 - Clinical priority ranking by false-negative rate.

    Ranks all ten models from lowest to highest false-negative rate.
    Lower is better: a lower FN-Rate means fewer diabetic patients are
    incorrectly told they are healthy. Bars are colour-coded:
      Green  - FN-Rate at or below 0.30 (good for screening)
      Amber  - FN-Rate between 0.31 and 0.42 (moderate)
      Red    - FN-Rate above 0.42 (not suitable for screening)

    This is the most clinically relevant figure in the thesis because
    it shows which model would actually be most useful for finding
    undiagnosed diabetic patients in a real health screening context.
    """
    fn_df    = results_df.sort_values("FN_Rate", ascending=True).reset_index(drop=True)
    fn_names = fn_df["Model"].tolist()
    fn_rates = fn_df["FN_Rate"].tolist()
    rec_vals = fn_df["Recall"].tolist()

    bar_colors = []
    for v in fn_rates:
        if v <= 0.30:   bar_colors.append(GREEN)
        elif v <= 0.42: bar_colors.append(AMBER)
        else:           bar_colors.append(RED)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    ax.xaxis.grid(True, linestyle="--", alpha=0.22, color=WHITE)
    ax.yaxis.grid(False)

    bars = ax.barh(fn_names, fn_rates, color=bar_colors,
                   edgecolor=WHITE, linewidth=0.6,
                   height=0.60, zorder=3)

    for bar, fnr, rec in zip(bars, fn_rates, rec_vals):
        ax.text(fnr + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"FN-Rate: {fnr:.3f}   |   Recall: {rec:.3f}",
                va="center", fontsize=9.5, color=WHITE, fontfamily=FONT)

    ax.set_xlabel("False-Negative Rate  (lower = clinically better)",
                  color=WHITE, fontsize=11, fontfamily=FONT)
    ax.set_title(
        "Figure 21.  Clinical Priority Ranking  -  "
        "False-Negative Rate Across All Ten Models",
        color=WHITE, fontsize=12, fontweight="bold",
        pad=14, fontfamily=FONT)
    ax.set_xlim(0, 0.74)
    ax.set_yticklabels(fn_names, color=WHITE, fontsize=10.5,
                       fontfamily=FONT)

    legend_patches = [
        mpatches.Patch(color=GREEN, label="FN-Rate at or below 0.30  -  Best for screening"),
        mpatches.Patch(color=AMBER, label="FN-Rate 0.31 to 0.42  -  Moderate"),
        mpatches.Patch(color=RED,   label="FN-Rate above 0.42  -  Not suitable for screening"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9.5,
              framealpha=0.42, facecolor=NAVY, labelcolor=WHITE)

    plt.tight_layout(pad=1.8)
    save_fig("Figure21_Clinical_FN_Rate_Ranking.png", fig)


def figure22_comprehensive(results_df: pd.DataFrame) -> None:
    """
    Figure 22 - Three-metric comprehensive comparison panel.

    Three bar charts arranged side by side covering all ten models:
      Panel 1 - Accuracy
      Panel 2 - ROC-AUC
      Panel 3 - False-Negative Rate

    This is the final summary figure used in the conclusions section.
    Showing all three metrics together makes it immediately visible
    that the model ranking changes depending on which metric you use,
    which is one of the key findings of the thesis.
    """
    df   = results_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    names  = df["Model"].tolist()
    acc    = df["Accuracy"].tolist()
    auc    = df["ROC_AUC"].tolist()
    fn     = df["FN_Rate"].tolist()
    xlbls  = [n.replace(" ", "\n") for n in names]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Figure 22.  Comprehensive Model Comparison  -  "
        "Accuracy  |  ROC-AUC  |  False-Negative Rate  (All 10 Models)",
        color=WHITE, fontsize=13, fontweight="bold",
        fontfamily=FONT, y=1.02)

    # Panel 1: Accuracy
    ax = axes[0]
    style_ax(ax)
    col_acc = [TEAL if n in ("ANN", "Random Forest", "Decision Tree")
               else BLUE for n in names]
    bars = ax.bar(x, acc, color=col_acc, edgecolor=WHITE,
                  lw=0.5, zorder=3, width=0.68)
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=WHITE, fontfamily=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=45, ha="right",
                       fontsize=8.5, color=WHITE)
    ax.set_ylabel("Accuracy", color=WHITE, fontsize=10.5, fontfamily=FONT)
    ax.set_title("Accuracy", color=WHITE, fontsize=12,
                 fontfamily=FONT, pad=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.75, color=AMBER, lw=1.2, ls="--", alpha=0.50)
    ax.text(-0.45, 0.757, "0.75", color=AMBER, fontsize=8)

    # Panel 2: ROC-AUC
    ax = axes[1]
    style_ax(ax)
    max_auc = max(auc)
    col_auc = [TEAL if v == max_auc else GREEN if v >= 0.80 else BLUE
               for v in auc]
    bars = ax.bar(x, auc, color=col_auc, edgecolor=WHITE,
                  lw=0.5, zorder=3, width=0.68)
    for bar, val in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=WHITE, fontfamily=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=45, ha="right",
                       fontsize=8.5, color=WHITE)
    ax.set_ylabel("ROC-AUC", color=WHITE, fontsize=10.5, fontfamily=FONT)
    ax.set_title("ROC-AUC  (higher is better)", color=WHITE,
                 fontsize=12, fontfamily=FONT, pad=10)
    ax.set_ylim(0.50, 0.96)
    ax.axhline(0.80, color=AMBER, lw=1.2, ls="--", alpha=0.50)
    ax.text(-0.45, 0.804, "0.80", color=AMBER, fontsize=8)

    # Panel 3: FN Rate
    ax = axes[2]
    style_ax(ax)
    col_fn = [GREEN if v <= 0.30 else AMBER if v <= 0.42 else RED
              for v in fn]
    bars = ax.bar(x, fn, color=col_fn, edgecolor=WHITE,
                  lw=0.5, zorder=3, width=0.68)
    for bar, val in zip(bars, fn):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.009,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=WHITE, fontfamily=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=45, ha="right",
                       fontsize=8.5, color=WHITE)
    ax.set_ylabel("False-Negative Rate", color=WHITE,
                  fontsize=10.5, fontfamily=FONT)
    ax.set_title("FN Rate  (lower is clinically better)", color=WHITE,
                 fontsize=12, fontfamily=FONT, pad=10)
    ax.set_ylim(0, 0.72)

    fn_legend = [
        mpatches.Patch(color=GREEN, label="At or below 0.30  -  Best"),
        mpatches.Patch(color=AMBER, label="0.31 to 0.42  -  Moderate"),
        mpatches.Patch(color=RED,   label="Above 0.42  -  Not suitable"),
    ]
    ax.legend(handles=fn_legend, loc="upper right", fontsize=8.8,
              framealpha=0.42, facecolor=NAVY, labelcolor=WHITE)

    plt.tight_layout(pad=2.0)
    save_fig("Figure22_Comprehensive_Comparison.png", fig)


def generate_all(dl_wrappers: dict, results_df: pd.DataFrame) -> None:
    """Generate and save Figures 17 through 22."""
    ann  = dl_wrappers["ANN"]
    dnn  = dl_wrappers["DNN"]
    lstm = dl_wrappers["LSTM"]

    print("  Figure 17 - ANN training curves")
    figure17_ann_curves(ann)
    print("  Figure 18 - DNN training curves")
    figure18_dnn_curves(dnn)
    print("  Figure 19 - LSTM training curves")
    figure19_lstm_curves(lstm)
    print("  Figure 20 - All DL models compared")
    figure20_dl_comparison(ann, dnn, lstm)
    print("  Figure 21 - Clinical FN-Rate ranking")
    figure21_clinical_ranking(results_df)
    print("  Figure 22 - Comprehensive 3-metric comparison")
    figure22_comprehensive(results_df)
