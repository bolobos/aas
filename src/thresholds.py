"""Rules that turn an anomaly score into an alarm threshold. Higher score = more anomalous.

| rule | needs | idea |
|:--|:--|:--|
| `quantile` | scores of normal validation data | alarm on the top `fpr` of normal behaviour (the false-alarm rate is chosen) |
| `sigma` / `mad` | scores of normal data | mean + k standard deviations, or median + k MADs (robust) |
| `isodata` | unlabelled scores, bimodal | iterate the mean (or median) of the two sides until the cut stops moving
  (Ridler and Calvard 1978; Velasco 1980; used on AE errors by Silva et al. 2025) |
| `otsu` | unlabelled scores, bimodal | the cut that maximises the between-class variance |
| `oracle_f1` | labels | upper bound only: it looks at the answer |
"""

import numpy as np


def quantile(scores: np.ndarray, fpr: float = 0.01) -> float:
    """Threshold with a false-alarm rate of `fpr` on the (normal) scores."""
    return float(np.quantile(scores, 1 - fpr))


def sigma(scores: np.ndarray, k: float = 3.0) -> float:
    return float(np.mean(scores) + k * np.std(scores))


def mad(scores: np.ndarray, k: float = 3.5) -> float:
    """median + k * 1.4826 * MAD: the Gaussian sigma estimated robustly."""
    m = np.median(scores)
    return float(m + k * 1.4826 * np.median(np.abs(scores - m)))


def isodata(scores: np.ndarray, center=np.mean, tol: float = 1e-9, max_iter: int = 200, log: bool = False) -> float:
    """ISODATA / intermeans threshold: start at the overall centre, then repeat T = (c(s <= T) + c(s > T)) / 2 with
    c = `np.mean` (Ridler-Calvard) or `np.median` (the variant of ASAP) until T stops moving. No labels are used, but the
    scores must be roughly bimodal: with a rare second mode the cut sits between the bulk and the tail.
    `log=True` works on log(score), which suits heavy-tailed errors."""
    s = np.log(np.maximum(scores, 1e-12)) if log else np.asarray(scores, dtype=float)
    t = float(center(s))
    for _ in range(max_iter):
        lo, hi = s[s <= t], s[s > t]
        if len(lo) == 0 or len(hi) == 0:
            break
        new = (center(lo) + center(hi)) / 2
        if abs(new - t) < tol:
            break
        t = float(new)
    return float(np.exp(t)) if log else t


def otsu(scores: np.ndarray, bins: int = 256, log: bool = False) -> float:
    """Otsu's threshold: the histogram cut that maximises the between-class variance."""
    s = np.log(np.maximum(scores, 1e-12)) if log else np.asarray(scores, dtype=float)
    h, edges = np.histogram(s, bins=bins)
    p, mid = h / h.sum(), (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu0 = np.cumsum(p * mid) / np.maximum(w0, 1e-12)
    mu1 = (np.sum(p * mid) - np.cumsum(p * mid)) / np.maximum(1 - w0, 1e-12)
    t = float(mid[np.argmax(w0 * (1 - w0) * (mu0 - mu1) ** 2)])
    return float(np.exp(t)) if log else t


def oracle_f1(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold with the best F1 on labelled scores: an upper bound, never a deployable rule."""
    order = np.argsort(scores)[::-1]
    ys = np.asarray(y)[order]
    tp = np.cumsum(ys)
    f1 = 2 * tp / (np.arange(1, len(ys) + 1) + ys.sum())
    return float(scores[order][np.argmax(f1)])
