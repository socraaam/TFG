"""
estilo_figuras.py — Estilo matplotlib y paleta compartidos por las figuras.
"""

import matplotlib.pyplot as plt

C_XGB = "#2E5EAA"
C_MLP = "#C25450"
C_BEN = "#7E8E83"
C_FAM = {"word_based": "#E08E45", "base32": "#4A7A8C", "encoded": "#7A4A7C", "benign": "#7E8E83"}


def aplicar_estilo():
    """Aplica el estilo comun (serif, sin spines superiores, grid tenue, dpi 150)."""
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
        "figure.dpi":       150,
    })
