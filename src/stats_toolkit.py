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

def z_test_two_means(x, y, alternative="two-sided"):
    """Large-sample Z-test for two means (sample variances used in place of sigma^2).
    z = (xbar - ybar) / sqrt(s1^2/n1 + s2^2/n2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    z = (x.mean() - y.mean()) / se
    if alternative == "two-sided":
        p = 2 * _st.norm.sf(abs(z))
    elif alternative == "greater":
        p = _st.norm.sf(z)
    else:
        p = _st.norm.cdf(z)
    return z, p


def z_test_two_proportions(k1, n1, k2, n2, alternative="two-sided"):
    """Two-proportion Z-test with the pooled proportion under H0: p1 = p2."""
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    if alternative == "two-sided":
        p = 2 * _st.norm.sf(abs(z))
    elif alternative == "greater":
        p = _st.norm.sf(z)
    else:
        p = _st.norm.cdf(z)
    return z, p

def regression_f_test(x, y):
    """Overall F-test for simple linear regression.
    SST = SSR + SSE;  F = (SSR / 1) / (SSE / (n - 2)),  df = (1, n - 2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    a, b = linear_fit(x, y)
    yhat = a + b * x
    sst = np.sum((y - y.mean()) ** 2)
    sse = np.sum((y - yhat) ** 2)
    ssr = sst - sse
    F = ssr / (sse / (n - 2))
    p = _st.f.sf(F, 1, n - 2)
    return {"SST": sst, "SSR": ssr, "SSE": sse, "n": n,
            "F": F, "p": p, "R2": ssr / sst}

def one_way_anova(groups):
    """One-way ANOVA from the formulas. groups: list of 1-D arrays.
    SST = SSB + SSW;  F = (SSB / (k-1)) / (SSW / (N-k));  eta^2 = SSB / SST."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    n_i = np.array([len(g) for g in groups])
    N = n_i.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])

    ssb = np.sum(n_i * (means - grand) ** 2)
    ssw = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    df_b, df_w = k - 1, N - k
    msb, msw = ssb / df_b, ssw / df_w
    F = msb / msw
    p = _st.f.sf(F, df_b, df_w)
    return {"SSB": ssb, "SSW": ssw, "SST": ssb + ssw,
            "df_between": df_b, "df_within": df_w,
            "MSB": msb, "MSW": msw, "F": F, "p": p,
            "eta2": ssb / (ssb + ssw)}

from itertools import combinations


def games_howell(groups, names):
    """Games-Howell pairwise comparisons (no equal-variance assumption).
    For each pair: t = (m_i - m_j) / sqrt(s_i^2/n_i + s_j^2/n_j), Welch-Satterthwaite df,
    q = sqrt(2) * |t|, p from the studentized range distribution with k groups."""
    k = len(groups)
    rows = []
    for i, j in combinations(range(k), 2):
        gi = np.asarray(groups[i], dtype=float)
        gj = np.asarray(groups[j], dtype=float)
        ni, nj = len(gi), len(gj)
        vi, vj = gi.var(ddof=1), gj.var(ddof=1)
        diff = gi.mean() - gj.mean()
        se2 = vi / ni + vj / nj
        t = diff / np.sqrt(se2)
        df = se2**2 / ((vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1))
        q = np.sqrt(2) * abs(t)
        p = _st.studentized_range.sf(q, k, df)
        rows.append({"group_i": names[i], "group_j": names[j],
                     "diff": diff, "t": t, "df": df, "q": q, "p": p})
    return rows

def chi_square_independence(table):
    """Chi-square test of independence on a contingency table of counts.
    E = row_total * col_total / N;  chi2 = sum((O - E)^2 / E);  df = (r-1)(c-1).
    Cramer's V = sqrt(chi2 / (N * (min(r, c) - 1))).
    Adjusted residual = (O - E) / sqrt(E * (1 - row/N) * (1 - col/N))."""
    O = np.asarray(table, dtype=float)
    N = O.sum()
    row = O.sum(axis=1, keepdims=True)
    col = O.sum(axis=0, keepdims=True)
    E = row @ col / N
    chi2 = np.sum((O - E) ** 2 / E)
    r, k = O.shape
    df = (r - 1) * (k - 1)
    p = _st.chi2.sf(chi2, df)
    v = np.sqrt(chi2 / (N * (min(r, k) - 1)))
    adj = (O - E) / np.sqrt(E * (1 - row / N) * (1 - col / N))
    return {"chi2": chi2, "df": df, "p": p, "cramers_v": v,
            "min_expected": E.min(), "adj_resid": adj, "N": N}

def chi_square_gof(observed, expected_props):
    """Chi-square goodness of fit: chi2 = sum((O - E)^2 / E), E = N * expected proportion,
    df = k - 1."""
    O = np.asarray(observed, dtype=float)
    E = O.sum() * np.asarray(expected_props, dtype=float)
    chi2 = np.sum((O - E) ** 2 / E)
    df = len(O) - 1
    return {"chi2": chi2, "df": df, "p": _st.chi2.sf(chi2, df), "min_expected": E.min()}

def chi_square_single_variance(x, sigma0_sq, alpha=0.05):
    """Chi-square test of H0: sigma^2 = sigma0^2 for one sample.
    chi2 = (n-1) s^2 / sigma0^2, df = n-1.  Two-sided p-value, plus a (1-alpha) CI for sigma^2."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s2 = x.var(ddof=1)
    df = n - 1
    chi2 = df * s2 / sigma0_sq
    p = min(1.0, 2 * min(_st.chi2.cdf(chi2, df), _st.chi2.sf(chi2, df)))
    ci = (df * s2 / _st.chi2.ppf(1 - alpha / 2, df),
          df * s2 / _st.chi2.ppf(alpha / 2, df))
    return {"n": n, "s2": s2, "chi2": chi2, "df": df, "p": p, "ci_var": ci}