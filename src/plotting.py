import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import to_rgba

class HandlerMultiSquare(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        colors = orig_handle              # tuple/list of color strings
        n = len(colors)
        w = width / n                     # each square gets 1/n of the width
        artists = []
        for i, c in enumerate(colors):
            r = Rectangle((xdescent + i*w, ydescent),
                          w, height,
                          transform=trans,
                          facecolor=c,
                          edgecolor='none')
            artists.append(r)
        return artists

def _rescale(
    data: np.ndarray,
    min_val: float,
    max_val: float,
    min_ratio: float = 0.50,
    max_ratio: float = 0.92,
) -> np.ndarray:
    """Rescale the data to a specified range [min_ratio, max_ratio]
    based on the given min_val and max_val."""
    data_rescaled = (data - min_val) / (max_val - min_val)
    data_rescaled = data_rescaled * (max_ratio - min_ratio) + min_ratio
    return data_rescaled

def plot_multi_bar(
    params_en: Dict[str, float],
    params_zh: Dict[str, float],
    max_param: float,
    min_param: float,
    xname: str,
    save_path: Path,
    title: str = "",
    legend: bool = True,
) -> None:
    """
    Plot the bar chart of parameters for English and Chinese model.
    
    Args:
        params_en (dict[str, float]): The parameter values for English model.
        params_zh (dict[str, float]): The parameter values for Chinese model.
        max_param (float): The maximum value for normalization.
        min_param (float): The minimum value for normalization.
        xname (str): The name of the x-axis.
        save_path (Path): The path to save the plot.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  
    plt.rcParams["font.size"] = 18
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    width = 0.4
    colors = ("#96ced3", "#e9c54e", "#e64b35")
    colors_alpha = tuple(to_rgba(c, alpha=0.3) for c in colors)
    x = np.arange(len(params_en))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # English results
    keys, values_en = zip(*params_en.items())
    norm_values_en = _rescale(np.array(values_en), min_param, max_param)
    rects_en = ax.bar(x + width, norm_values_en, width, label="English", color=colors)
    values_en = [f"{v:.4f}" for v in values_en]
    ax.bar_label(rects_en, labels=values_en, padding=3, fmt="%.4f")
    # Chinese results
    keys, values_zh = zip(*params_zh.items())
    norm_values_zh = _rescale(np.array(values_zh), min_param, max_param)
    rects_zh = ax.bar(x, norm_values_zh, width, label="Chinese", color=colors_alpha)
    values_zh = [f"{v:.4f}" for v in values_zh]
    ax.bar_label(rects_zh, labels=values_zh, padding=3, fmt="%.4f")
    
    ax.set_xticks(x + width / 2, keys)
    ax.set_yticks([])
    ax.set_xlabel(xname)
    ax.set_ylim(0.0, 1.0)
    if legend:
        ax.legend(
            [colors, colors_alpha],
            ["English", "Chinese"],
            handler_map={tuple: HandlerMultiSquare()},
            loc='upper right', bbox_to_anchor=(1.3, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)