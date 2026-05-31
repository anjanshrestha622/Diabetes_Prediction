"""
src/utils.py
============
Small helper functions used across all other modules.

Three functions are defined here:
  banner(text)         - prints a section header to the console
  save_fig(name, fig)  - saves a matplotlib figure to the figures folder
  style_ax(ax)         - applies the consistent dark theme to an axes object

Putting these here means every other file stays focused on its main job
and the visual style stays identical across all 22 figures automatically.
"""

import os
import matplotlib.pyplot as plt
from config import FIGURES_DIR, RESULTS_DIR, DPI, DARK_BG, WHITE


def make_output_dirs():
    """Create the output folders if they do not already exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def banner(text: str) -> None:
    """
    Print a clearly visible section header to the console.
    This makes it easy to follow which phase of the pipeline is running.

    Example output:
    ================================================================
      PHASE 2 - CLASSICAL MACHINE LEARNING MODELS
    ================================================================
    """
    line = "=" * 66
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def save_fig(filename: str, fig: plt.Figure) -> None:
    """
    Save a matplotlib figure to the figures output folder, then close it.

    The figure is saved at the resolution set in config.py (200 dpi by
    default) so that all figures are sharp enough for a printed thesis.
    Closing after saving frees up memory for the next figure.

    Parameters
    ----------
    filename : the name to save the file as, e.g. 'Figure5_Heatmap.png'
    fig      : the matplotlib Figure object to save
    """
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"    Saved  ->  {filename}")


def style_ax(ax: plt.Axes, bg: str = DARK_BG) -> None:
    """
    Apply the dark theme to a single matplotlib Axes object.

    This sets the background colour, tick colour, spine colour,
    and adds a light horizontal grid. Calling this on every axes
    before adding data ensures all figures look consistent.

    Parameters
    ----------
    ax : the matplotlib Axes to style
    bg : background hex colour (defaults to the dark background set in config)
    """
    ax.set_facecolor(bg)
    ax.tick_params(colors=WHITE, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#3A3A4A")
    ax.yaxis.grid(True, linestyle="--", alpha=0.20, color=WHITE, zorder=0)
    ax.set_axisbelow(True)
