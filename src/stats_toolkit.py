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

def linear_fit(x, y):
    """Least-squares line y = a + b*x from the normal equations.

    b = sum((x - xbar)(y - ybar)) / sum((x - xbar)^2),  a = ybar - b * xbar
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    b = np.sum(dx * (y - y.mean())) / np.sum(dx**2)
    a = y.mean() - b * x.mean()
    return a, b

def two_regression_lines(x, y):
    """Both regression lines and the link to r.

    b_yx = slope of y on x, b_xy = slope of x on y.  b_yx * b_xy = r^2.
    """
    a_yx, b_yx = linear_fit(x, y)
    a_xy, b_xy = linear_fit(y, x)
    return {"b_yx": b_yx, "b_xy": b_xy, "r2_from_slopes": b_yx * b_xy,
            "r_from_slopes": np.sign(b_yx) * np.sqrt(b_yx * b_xy)}

def average_ranks(x):
    """Ranks 1..n, with tied values given the average of their ranks."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman_rho(x, y):
    """Spearman rank correlation = Pearson r of the (tie-averaged) ranks."""
    return pearson_r(average_ranks(x), average_ranks(y))

def polynomial_fit(x, y, degree):
    """Least-squares polynomial via the normal equations (X'X) c = X'y.
    Returns coefficients [a0, a1, ..., a_degree]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.vander(x, degree + 1, increasing=True)
    return np.linalg.solve(X.T @ X, X.T @ y)


def poly_predict(coefs, x):
    x = np.asarray(x, dtype=float)
    return sum(c * x**k for k, c in enumerate(coefs))


def r_squared(y, yhat):
    """R^2 = 1 - SS_residual / SS_total."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

def kendall_tau_b(x, y):
    """Kendall's tau-b: (concordant - discordant) / sqrt((n0 - n1)(n0 - n2)),
    where n1 and n2 count tied pairs in x and in y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    s = np.sum(sx * sy) / 2                    # concordant minus discordant pairs
    n0 = n * (n - 1) / 2
    n1 = (np.sum(sx == 0) - n) / 2             # pairs tied in x
    n2 = (np.sum(sy == 0) - n) / 2             # pairs tied in y
    return s / np.sqrt((n0 - n1) * (n0 - n2))

from scipy import stats as _st


def two_sample_t(x, y, equal_var=False, alternative="two-sided"):
    """Two-sample t-test from the formulas.

    equal_var=False -> Welch (separate variances); True -> pooled variance.
    alternative: 'two-sided', 'greater' (mean x > mean y), or 'less'.
    Returns t, df, p. Uses scipy only for the t distribution's tail areas.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    t = (m1 - m2) / se
    if alternative == "two-sided":
        p = 2 * _st.t.sf(abs(t), df)
    elif alternative == "greater":
        p = _st.t.sf(t, df)
    else:
        p = _st.t.cdf(t, df)
    return t, df, p


def cohens_d(x, y):
    """Standardized mean difference using the pooled standard deviation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp

def f_test_variances(x, y, alternative="two-sided"):
    """F-test for equality of two variances: F = s1^2 / s2^2 with (n1-1, n2-1) df.
    H0: sigma1^2 = sigma2^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    F = x.var(ddof=1) / y.var(ddof=1)
    df1, df2 = len(x) - 1, len(y) - 1
    if alternative == "two-sided":
        p = min(1.0, 2 * min(_st.f.cdf(F, df1, df2), _st.f.sf(F, df1, df2)))
    elif alternative == "greater":
        p = _st.f.sf(F, df1, df2)
    else:
        p = _st.f.cdf(F, df1, df2)
    return F, df1, df2, p