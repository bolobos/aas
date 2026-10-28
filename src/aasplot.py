"""Plot helper shared by every demo notebook: one figure is drawn once, shown in the notebook and stored as the slide
figure (a transparent PDF in `assets/figures/plots/<module>/`).

    import aasplot as ap
    ap.setup("03")                      # once, in the first cell: deck style and the module folder
    f, ax = ap.fig("full", 1.6)         # size = size on the slide (text width 4.3 in, a column 2.15 in)
    ax.plot(...)
    ap.save(f, "name")                  # writes name.pdf at slide size, then shows a larger copy in the notebook

Colours are the ones of `slides/header.tex`. Text is placed by matplotlib inside the figure bounds, so it cannot spill
over a slide.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]  # repo root: this file lives in src/
BLUE, RED, GREEN, GOLD, GREY = "#1F618D", "#B03A2E", "#1E8449", "#B9770E", "#5D6D7E"
LIGHT = "#AEB6BF"
PALETTE = [BLUE, RED, GREEN, GOLD, GREY]
FULL, COL = 4.3, 2.15  # slide text width and one column of a two-column frame, inches
_module = {"name": ""}


def setup(module: str, dpi: int = 130) -> None:
    """Deck style (Noto Sans, small text, light grid, no top/right spines) and the output folder `module`
    (01 intro, 02 spam, 03 anomaly, 04 malware). The notebook shows the figure at the size it has on the slide."""
    _module["name"] = module
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.3, "legend.frameon": False, "axes.grid": True, "grid.color": "#D5D8DC",
        "grid.linewidth": 0.4, "axes.axisbelow": True, "savefig.transparent": True, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans", "figure.dpi": dpi, "figure.facecolor": "white",
    })  # fmt: skip
    sns.set_palette(PALETTE)


def fig(kind: str = "full", height: float = 1.6, **kw):
    """Figure at its slide size: kind "full" (text width) or "col" (a column of a two-column frame); extra keywords go
    to `plt.subplots` (ncols, sharey, gridspec_kw ...)."""
    return plt.subplots(figsize=(FULL if kind == "full" else COL, height), **kw)


def save(f, name: str, show: bool = True, scale: float = 1.8) -> None:
    """Two independent operations on one figure: (1) write `assets/figures/plots/<module>/<name>.pdf` at the slide size
    (transparent, tight, silent); (2) enlarge the figure by `scale` and show it in the notebook at a readable size."""
    out = ROOT / "assets" / "figures" / "plots" / _module["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    f.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.03)
    if show:
        w, h = f.get_size_inches()
        f.set_size_inches(w * scale, h * scale)
        f.set_facecolor("white")
        plt.show()
    plt.close(f)
    return None


def grouped_bars(ax, groups, series, ylabel: str = "", colors=PALETTE, width: float = 0.8, ymax: float = 1.12):
    """Bars of several series side by side, one group per tick; `series` is a list of (name, values) and every bar
    carries its value."""
    n = len(series)
    for i, (name, vals) in enumerate(series):
        pos = np.arange(len(groups)) + (i - (n - 1) / 2) * width / n
        for b, v in zip(ax.bar(pos, vals, width / n, color=colors[i], label=name, lw=0), vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.2f}".lstrip("0") if v < 1 else "1",
                ha="center",
                fontsize=6.5,
            )
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set(ylabel=ylabel, ylim=(0, ymax))
    ax.grid(axis="x", visible=False)


def hbars(ax, labels, values, color=BLUE, fmt: str = "{:.3f}", xlabel: str = "", xmax: float | None = None):
    """Horizontal bars with their values, best on top (sorted by value)."""
    order = np.argsort(values)
    vals = np.asarray(values, dtype=float)[order]
    ax.barh([labels[i] for i in order], vals, color=color, height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + 0.008 * (xmax or vals.max()), i, fmt.format(v), va="center", fontsize=7)
    ax.set(xlabel=xlabel, xlim=(0, xmax or vals.max() * 1.15))
    ax.grid(axis="y", visible=False)


def ellipse(mu, cov, d: float, n: int = 120) -> np.ndarray:
    """Points at Mahalanobis distance d from mu."""
    w, v = np.linalg.eigh(np.asarray(cov))
    t = np.linspace(0, 2 * np.pi, n)
    return (np.asarray(mu)[:, None] + v @ (d * np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)]))).T
