"""
src/visualization/eda_figures.py
================================
Generates the three exploratory data analysis figures:

  Figure 4 - Dataset information panel (df.info style summary)
  Figure 5 - Correlation heatmap showing feature associations
  Figure 6 - Feature distributions before and after preprocessing

These figures support the Experiments section of the thesis by showing
what the dataset looks like, how the features relate to each other,
and what the preprocessing step changes in the data distributions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpatches
import matplotlib.lines     as mlines
from   matplotlib.patches   import FancyBboxPatch
import seaborn as sns

from config import (DARK_BG, NAVY, WHITE, GREY, FONT, FEATURES,
                    ZERO_COLS, BLUE, RED, TEAL)
from src.utils import save_fig, style_ax


def figure4_dataset_info(df_raw: pd.DataFrame) -> None:
    """
    Figure 4 - Dataset structure panel.

    Shows column names, data types, non-null counts, class balance,
    and memory usage in a terminal-style dark panel. This figure
    corresponds to running df.info() on the raw dataset.
    """
    fig = plt.figure(figsize=(12, 9), facecolor=NAVY)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=NAVY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Title bar
    ax.add_patch(FancyBboxPatch(
        (0.03, 0.91), 0.94, 0.08,
        boxstyle="round,pad=0.01", facecolor="#0a0a20", lw=0))
    ax.text(0.5, 0.952, "PIMA Indian Diabetes Dataset  -  df.info() Summary",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=WHITE, fontfamily=FONT)

    # Sub-header lines
    sub = [
        "<class 'pandas.core.frame.DataFrame'>",
        f"RangeIndex:  {len(df_raw)} entries,  0 to {len(df_raw)-1}",
        f"Data columns:  (total {len(df_raw.columns)} columns)",
    ]
    for i, line in enumerate(sub):
        ax.text(0.06, 0.875 - i * 0.033, line,
                fontsize=9.5, color="#99AACC",
                fontfamily="monospace", va="center")

    # Column headers
    hxs = [0.06, 0.14, 0.60, 0.82]
    for hx, h in zip(hxs, ["#", "Column", "Non-Null Count", "Dtype"]):
        ax.text(hx, 0.768, h, fontsize=9.5, fontweight="bold",
                color="#7EC8F5", fontfamily="monospace", va="center")

    ax.plot([0.04, 0.96], [0.752, 0.752], color="#7EC8F5", lw=0.8, alpha=0.7)

    # Data rows
    for i, col in enumerate(df_raw.columns):
        y     = 0.712 - i * 0.065
        bg    = "#14172A" if i % 2 == 0 else "#0e1022"
        ax.add_patch(FancyBboxPatch(
            (0.04, y - 0.030), 0.92, 0.056,
            boxstyle="square,pad=0", facecolor=bg, lw=0))
        nn    = df_raw[col].notna().sum()
        dtype = str(df_raw[col].dtype)
        for hx, val, vc in zip(hxs,
                                [str(i), col, f"{nn} non-null", dtype],
                                ["#F39C12", "#F1C40F", "#2ECC71", "#E74C3C"]):
            ax.text(hx, y, val, fontsize=9.2, color=vc,
                    fontfamily="monospace", va="center")

    bot_y = 0.712 - len(df_raw.columns) * 0.065 - 0.012
    ax.plot([0.04, 0.96], [bot_y, bot_y],
            color="#7EC8F5", lw=0.7, alpha=0.5)

    int_c   = sum(1 for c in df_raw.columns if df_raw[c].dtype == "int64")
    float_c = sum(1 for c in df_raw.columns if df_raw[c].dtype == "float64")
    mem_kb  = df_raw.memory_usage(deep=True).sum() / 1024
    nd_n    = (df_raw["Outcome"] == 0).sum()
    d_n     = (df_raw["Outcome"] == 1).sum()

    footer = [
        (f"dtypes:  int64({int_c}),  float64({float_c})", "#E74C3C"),
        (f"memory usage:  {mem_kb:.1f} KB",               "#2ECC71"),
        (f"Class balance  :  Non-Diabetic = {nd_n} (65.1%)   "
         f"Diabetic = {d_n} (34.9%)",                     "#7EC8F5"),
    ]
    fy = bot_y - 0.040
    for txt, col in footer:
        ax.text(0.06, fy, txt, fontsize=9.0, color=col,
                fontfamily="monospace", va="center")
        fy -= 0.044

    ax.text(0.5, 0.018,
            "Figure 4.  Dataset Information Panel  "
            "(PIMA Indian Diabetes Dataset - Structure and Class Distribution)",
            ha="center", va="center", fontsize=9.5,
            color=GREY, style="italic", fontfamily=FONT)

    save_fig("Figure4_Dataset_Info.png", fig)


def figure5_correlation_heatmap(df_clean: pd.DataFrame) -> None:
    """
    Figure 5 - Correlation heatmap.

    Shows pairwise Pearson correlations between all eight features
    and the Outcome variable. The Outcome row and column are
    highlighted with a dark border to draw attention to the
    feature-to-target relationships that matter most for prediction.
    """
    corr = df_clean[FEATURES + ["Outcome"]].corr()

    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    fig.patch.set_facecolor("white")

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap=cmap,
        center=0, vmin=-0.7, vmax=0.7, square=True,
        linewidths=0.6, linecolor="#E0E0E0",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.80, "label": "Pearson r"},
    )

    # Highlight the Outcome column and row
    n   = len(corr.columns)
    oi  = list(corr.columns).index("Outcome")
    for rect in [
        mpatches.Rectangle((oi, 0), 1, n, fill=False,
                            edgecolor="#1a1a2e", lw=2.8, clip_on=False),
        mpatches.Rectangle((0, oi), n, 1, fill=False,
                            edgecolor="#1a1a2e", lw=2.8, clip_on=False),
    ]:
        ax.add_patch(rect)

    ax.set_title(
        "Correlation Heatmap  -  PIMA Indian Diabetes Dataset\n"
        "(border marks the Outcome variable row and column)",
        fontsize=12, fontweight="bold", pad=16, color="#1a1a2e")
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=40, ha="right", fontsize=9.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)

    ax.text(0.5, -0.13,
            "Glucose has the strongest positive correlation with Outcome "
            "(r = 0.49).  BMI, Age, and Pregnancies follow as secondary predictors.",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#333333", style="italic")

    fig.text(0.5, -0.04,
             "Figure 5.  Correlation Heatmap  "
             "(Pairwise Feature Associations with the Diabetes Outcome Variable)",
             ha="center", fontsize=10, color="#333333", style="italic")

    plt.tight_layout(pad=1.8)
    save_fig("Figure5_Correlation_Heatmap.png", fig)


def figure6_feature_distributions(df_raw:     pd.DataFrame,
                                   df_clean:   pd.DataFrame,
                                   medians:    dict) -> None:
    """
    Figure 6 - Feature distributions.

    Eight-panel grid showing the distribution of each feature
    separately for diabetic and non-diabetic patients. Solid filled
    bars show the raw data. Dashed outlines show the distribution
    after median imputation. Vertical lines mark the median for each
    class. This lets you see at a glance which features separate
    the two classes most clearly.
    """
    c_nd = "#2E74B5"  # non-diabetic colour
    c_d  = "#e74c3c"  # diabetic colour
    BINS = 24

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "PIMA Indian Diabetes  -  Feature Distributions\n"
        "Blue = Non-Diabetic (n=500)   |   Red = Diabetic (n=268)   |   "
        "Dashed outline = After Median Imputation",
        fontsize=12, fontweight="bold", color="#1a1a2e", y=1.01)

    gs = gridspec.GridSpec(2, 4, hspace=0.54, wspace=0.38, figure=fig)

    for idx, feat in enumerate(FEATURES):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.set_facecolor("#F8F9FA")
        for sp in ax.spines.values():
            sp.set_color("#CCCCCC")

        nd_raw = df_raw[feat][df_raw["Outcome"] == 0]
        d_raw  = df_raw[feat][df_raw["Outcome"] == 1]
        nd_cl  = df_clean[feat][df_clean["Outcome"] == 0]
        d_cl   = df_clean[feat][df_clean["Outcome"] == 1]

        # Raw filled histograms
        ax.hist(nd_raw, bins=BINS, alpha=0.38, color=c_nd, density=True)
        ax.hist(d_raw,  bins=BINS, alpha=0.38, color=c_d,  density=True)

        # After-imputation outlines
        ax.hist(nd_cl, bins=BINS, alpha=0, color=c_nd,
                histtype="step", lw=1.8, ls="--", density=True)
        ax.hist(d_cl,  bins=BINS, alpha=0, color=c_d,
                histtype="step", lw=1.8, ls="--", density=True)

        # Median lines
        ax.axvline(nd_cl.median(), color=c_nd, lw=1.6, alpha=0.9)
        ax.axvline(d_cl.median(),  color=c_d,  lw=1.6, alpha=0.9)

        ax.set_title(feat, fontsize=10, fontweight="bold",
                     color="#1a1a2e", pad=4)
        ax.set_xlabel("Value", fontsize=8.5, color="#444444")
        ax.set_ylabel("Density", fontsize=8.5, color="#444444")
        ax.tick_params(labelsize=8)

        # Note imputed features
        if feat in medians:
            ax.text(0.97, 0.97, f"Med={medians[feat]}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="#888888", style="italic")

        # Mean annotation box
        ax.text(0.03, 0.97,
                f"ND  mean={nd_cl.mean():.1f}\nD   mean={d_cl.mean():.1f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.2, color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.78, edgecolor="#CCCCCC"))

    legend_handles = [
        mpatches.Patch(facecolor=c_nd, alpha=0.55,
                       label="Non-Diabetic (Outcome=0, n=500)"),
        mpatches.Patch(facecolor=c_d,  alpha=0.55,
                       label="Diabetic  (Outcome=1, n=268)"),
        mlines.Line2D([], [], color=c_nd, lw=1.8,
                      label="Non-Diabetic Median"),
        mlines.Line2D([], [], color=c_d,  lw=1.8,
                      label="Diabetic Median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.025))

    fig.text(0.5, -0.05,
             "Figure 6.  Feature Distributions  "
             "(Solid fill = raw data   |   Dashed outline = after imputation)",
             ha="center", fontsize=9.5, color="#333333", style="italic")

    save_fig("Figure6_Feature_Distributions.png", fig)


def generate_all(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                 medians: dict) -> None:
    """Generate and save Figures 4, 5, and 6."""
    print("  Figure 4 - Dataset info panel")
    figure4_dataset_info(df_raw)
    print("  Figure 5 - Correlation heatmap")
    figure5_correlation_heatmap(df_clean)
    print("  Figure 6 - Feature distributions")
    figure6_feature_distributions(df_raw, df_clean, medians)
