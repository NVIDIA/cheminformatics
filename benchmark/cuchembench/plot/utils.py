import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional

# PoR acceptance criteria
# ACCEPTANCE_CRITERIA = {'validity': 0.98, 'novelty': 0.50} # TODO


def setup_plot_grid(num_plots: int, 
                plots_per_row: int, 
                xscale: int = 7, 
                yscale: int = 4):
    """Setup plotting axes"""

    rows = int(math.ceil(num_plots / plots_per_row))

    fig = plt.figure(figsize=(plots_per_row * xscale, (rows * yscale)), facecolor=(1.0, 1.0, 1.0))
    axes_list = fig.subplots(rows, plots_per_row)
    if num_plots > 1:
        axes_list = axes_list.flatten()
    else:
        axes_list = [axes_list]

    for ax in axes_list[num_plots:]:
        fig.delaxes(ax)

    return fig, axes_list


def set_plotting_style(show_grid: bool = False):
    """Seaborn plotting style setup"""
    # Palette config
    sns.set_palette('dark')
    pal = sns.color_palette()
    sns.set_palette([pal[0]] + pal[2:])
    
    # Axes config
    if show_grid:
        kwargs = {'axes.edgecolor': 'black', 'axes.linewidth': 1.5}
    else:
        kwargs = {'axes.grid' : False}
    sns.set_style("whitegrid", kwargs)


def label_bars(ax: plt.Axes, 
                max_value: Optional[float] = None):
    """Add value labels to all bars in a bar plot"""
    for p in ax.patches:
        value = p.get_height()

        if value < 0:
            va = 'top'
        else:
            va = 'bottom'

        if not math.isclose(value, 0.0):
            label = "{:.2f}".format(value)
            x, y = p.get_x() * 1.005, value * 1.005

            bbox = None
            if max_value:
                if y >= max_value:
                    bbox = dict(boxstyle="square", fc=(1.0, 1.0, 1.0), ec=(0.0, 0.0, 0.0))
                y = min(y, max_value)                
            ax.annotate(label, (x, y), va=va, bbox=bbox)
