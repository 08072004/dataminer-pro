# ============================================================
#  utils/plots.py  —  Fonctions utilitaires pour les graphiques
#                     (thème sombre matplotlib)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np


def dark_fig(figsize=(10, 5)):
    """
    Crée une figure matplotlib avec le thème sombre de l'application.

    Retourne : (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#161b22")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return fig, ax


def dark_fig_multi(nrows=1, ncols=2, figsize=(14, 5)):
    """
    Crée une figure multi-axes avec le thème sombre.

    Retourne : (fig, axes)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="#161b22")
    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in ax_list:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    return fig, axes