import numpy as np


def pearson_r(x, y):
    """Pearson correlation coefficient, computed from the formula.

    r = sum((x - xbar) * (y - ybar)) / sqrt(sum((x - xbar)^2) * sum((y - ybar)^2))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")

    dx = x - x.mean()
    dy = y - y.mean()
    return np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2))

def correlation_summary(x, y):
    """r, r^2, standard error, probable error and a 95% Fisher-z interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = pearson_r(x, y)

    se = (1 - r**2) / np.sqrt(n)        # textbook SE of r
    pe = 0.6745 * se                     # probable error

    z = np.arctanh(r)                    # Fisher z transform
    z_se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * z_se), np.tanh(z + 1.96 * z_se)

    return {"n": n, "r": r, "r2": r**2, "se": se, "pe": pe, "ci95": (lo, hi)}