import math
import matplotlib.pyplot as plt
import pandas as pd


# PoR acceptance criteria
ACCEPTANCE_CRITERIA = {'validity': 0.98, 'novelty': 0.50}


def _label_bars(ax, max_value=None):
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
