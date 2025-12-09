import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

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


def plot_multi_bar_v2(
    params_list: List[Dict[str, float]],
    params_names: List[str],
    max_param: float,
    min_param: float,
    xname: str,
    save_path: Path,
    yname: str = "",
    title: str = "",
    legend: bool = True,
) -> None:
    """
    Plot the bar chart of parameters for different evaluation languages.
    
    Args:
        params_list (List[Dict[str, float]]): The parameter values for different models.
        params_names (List[str]): The names of the parameters.
        max_param (float): The maximum value for normalization.
        min_param (float): The minimum value for normalization.
        xname (str): The name of the x-axis.
        save_path (Path): The path to save the plot.
        yname (str): The name of the y-axis.
        title (str): The title of the plot.
    """
    plt.rcParams["font.family"] = "arial"  
    plt.rcParams["font.size"] = 18
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'
    
    L = len(params_list)
    width = 0.9 / L * 2
    colors = ("#96ced3", "#e9c54e", "#e64b35", "#8491b4")
    # colors_alpha = tuple(to_rgba(c, alpha=0.3) for c in colors)
    # colors_alpha_2 = tuple(to_rgba(c, alpha=0.6) for c in colors)
    x = np.arange(len(params_list[0])) * 2
    
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, params in enumerate(params_list):
        keys, values = zip(*params.items())
        norm_values = _rescale(np.array(values), min_param, max_param)
        rects = ax.bar(x + (i - L / 2 + 1 / 2) * width, norm_values, width, label=params_names[i], color=colors[i % len(colors)])
        values = [f"{v:.4f}" for v in values]
        ax.bar_label(rects, labels=values, padding=3, fmt="%.4f")
    ax.set_xticks(x, keys)
    ax.set_yticks([])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_ylim(0.0, 1.0)
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_length_distribution(
    token_lengths: List[List[int]],
    save_path: Path,
    xname: str,
    title: str = "",
    legend: bool = True,
    legend_labels: List[str] = None,
    smooth: bool = True,
    plot_avg: bool = True,
) -> None:
    plt.rcParams["font.family"] = "arial"  
    plt.rcParams["font.size"] = 18
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'arial'
    # plt.rcParams['mathtext.it'] = 'arial:italic'
    # plt.rcParams['mathtext.bf'] = 'arial:bold'

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#80ae9a", "#f6d6c2", "#df92d9", "#c1ae3d"]
    outlines_colors = ["#568b87", "#d47264", "#bf24b3", "#f5e582"]

    for i, token_length in enumerate(token_lengths):
        token_length = np.array(
            [L for L in token_length if 0 < L < 15000]
        )

        # Histogram (set density=True so KDE is on the same scale)
        ax.hist(
            token_length,
            bins=50,
            alpha=0.5,
            density=True,
            color=colors[i % len(colors)],
            label=legend_labels[i] if legend_labels is not None else None,
        )

        if smooth and len(token_length) > 1:
            kde = gaussian_kde(token_length, bw_method=0.05)
            xs = np.linspace(token_length.min(), token_length.max(), 500)
            ys = kde(xs)

            # Smooth outline
            ax.plot(xs, ys, lw=5, color=outlines_colors[i % len(outlines_colors)], linestyle="-")

            # “Enclosed outline” – fill under the curve (optional)
            # ax.fill_between(xs, ys, alpha=0.15, color=colors[i % len(colors)])

    if plot_avg:
        y_max = ax.get_ylim()[1]
        for i, token_length in enumerate(token_lengths):
            avg = np.mean(token_length)
            ax.vlines(avg, 0, y_max, color=outlines_colors[i % len(outlines_colors)], linestyle="--", lw=5)
                

    ax.set_xlabel("Output Lengths" if xname is None else xname)
    ax.set_yticks([])

    if legend:
        ax.legend(loc='upper right')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    

def plot_length_distribution_ridge(
    token_lengths: List[List[int]],
    save_path: Path,
    xname: str = "",
    title: str = "",
    legend: bool = True,
    legend_labels: List[str] = None,
    plot_avg: bool = True,
) -> None:
    """
    Ridge / joyplot style visualization of output-length distributions.
    One shared x-axis; curves stacked vertically with slight overlap.
    """

    plt.rcParams["font.family"] = "arial"
    plt.rcParams["font.size"] = 18

    # Filter and collect all data first
    cleaned = []
    for tl in token_lengths:
        arr = np.array([L for L in tl if 0 < L < 15000])
        if len(arr) > 1:
            cleaned.append(arr)
        else:
            cleaned.append(None)

    # global x-range
    all_vals = np.concatenate([c for c in cleaned if c is not None])
    x_min, x_max = 0, 5000   # Fix
    xs = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#80ae9a", "#f6d6c2", "#df92d9", "#c1ae3d"]
    n = len(token_lengths)
    offset_step = 1.2  # vertical distance between ridges

    max_height = 0.0
    kdes = []
    for c in cleaned:
        if c is None:
            kdes.append(None)
            continue
        kde = gaussian_kde(c, bw_method=0.05)  # tweak 0.25 for smoothness
        ys = kde(xs)
        kdes.append(ys)
        max_height = max(max_height, ys.max())

    # Normalize heights so they are comparable and fit nicely
    if max_height == 0:
        max_height = 1.0

    for i, (c, ys) in enumerate(zip(cleaned, kdes)):
        if c is None or ys is None:
            continue

        color = colors[i % len(colors)]
        offset = i * offset_step
        ys_norm = ys / max_height  # normalize to [0, ~1]

        # Ridge fill + outline
        ax.fill_between(xs, offset, offset + ys_norm, color=color, alpha=0.6)
        ax.plot(xs, offset + ys_norm, color=color, lw=4)

        # Mean line for that language
        mean_val = c.mean()
        ax.vlines(mean_val, offset, offset + ys_norm.max(), color=color,
                  linestyles="--", linewidth=4, alpha=0.8)

        # Label on the left
        if legend_labels is not None:
            ax.text(x_min - 0.03 * (x_max - x_min),
                    offset + 0.5 * ys_norm.max(),
                    legend_labels[i],
                    ha="right", va="center")

    ax.set_xlabel(xname if xname else "Output Lengths")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlim(x_min, x_max)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)