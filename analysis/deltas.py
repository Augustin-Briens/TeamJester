"""Phase 5 - Batch_1 and Batch_2 expressed as deltas from the Batch_3 baseline.

For every feature and each of B1/B2 vs B3:
    absolute delta, % delta, standardised delta (in B3 std), bootstrap 95% CI
    on the mean difference, Mann-Whitney p, Holm-corrected p across all
    features, classification {different, suggestive, equivalent,
    undetermined}, and the smallest difference this dataset could detect
    (bootstrap CI half-width).
Writes tables/deltas.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import common as C


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(np.nan_to_num(p, nan=1.0))
    adj = np.empty_like(p)
    m = len(p)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = max(prev, (m - rank) * np.nan_to_num(p[idx], nan=1.0))
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj


def run(F_, kept, exclude_ids=()):
    full = F_[F_.subset == "full"].copy()
    b3 = full[(full.batch == C.BASELINE) & (~full.image_id.isin(exclude_ids))]
    rows = []
    rng = np.random.default_rng(0)
    for batch in ("Batch_1", "Batch_2"):
        bb = full[(full.batch == batch) & (~full.image_id.isin(exclude_ids))]
        for f in kept:
            x = bb[f].astype(float).dropna().values
            y = b3[f].astype(float).dropna().values
            if len(x) < 3 or len(y) < 3:
                continue
            d = x.mean() - y.mean()
            pct = 100 * d / abs(y.mean()) if y.mean() != 0 else np.nan
            sd = y.std(ddof=1)
            sdelt = d / sd if sd > 0 else np.nan
            ci = C.bootstrap_ci_diff(x, y, rng=rng)
            try:
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            except Exception:
                p = np.nan
            half = (ci[1] - ci[0]) / 2
            rows.append(dict(batch=batch, feature=f,
                             mean_batch=x.mean(), mean_b3=y.mean(),
                             delta=d, pct_delta=pct, std_delta=sdelt,
                             ci_lo=ci[0], ci_hi=ci[1], p_mw=p,
                             mdd=half,  # smallest detectable ~ CI half-width
                             n_batch=len(x), n_b3=len(y)))
    D = pd.DataFrame(rows)
    # Holm within each batch comparison
    for b in ("Batch_1", "Batch_2"):
        m = D.batch == b
        D.loc[m, "p_holm"] = holm(D.loc[m, "p_mw"].values)
    def classify(r):
        inside = (r.ci_lo < 0 < r.ci_hi)
        narrow = (r.ci_lo > -0.10 * abs(r.mean_b3)) and \
                 (r.ci_hi < 0.10 * abs(r.mean_b3))
        if (not inside) and r.p_holm < 0.05:
            return "different"
        if not inside:
            return "suggestive"
        if inside and narrow:
            return "equivalent"
        return "undetermined"
    D["classification"] = D.apply(classify, axis=1)
    D["abs_std_delta"] = D["std_delta"].abs()
    D = D.sort_values(["batch", "abs_std_delta"], ascending=[True, False])
    D.to_csv(os.path.join(C.TABLE_DIR, "deltas.csv"), index=False)
    C.log(f"deltas done: {len(D)} rows")
    return D


if __name__ == "__main__":
    import features as FT
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    rep = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
    kept, _ = FT.prune(F_, rep)
    run(F_, kept)
