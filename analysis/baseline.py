"""Phase 3 - Batch_3 baseline statistics and the 'normal envelope'.

For every retained feature: mean, median, std, 5-95 percentile range and a
bootstrap 95% CI on the mean (resampling images). Checks internal
consistency (outlier images, sub-groups) and defines the per-feature and
joint normal envelope used by categorise.py.

Writes: tables/baseline_stats.csv, tables/envelope.json
"""
import os
import json
import numpy as np
import pandas as pd
import common as C


def run(F_, kept, exclude_ids=()):
    full = F_[F_.subset == "full"].copy()
    base = full[(full.batch == C.BASELINE) &
                (~full.image_id.isin(exclude_ids))]
    rows = []
    for f in kept:
        v = base[f].astype(float)
        lo95, hi95 = C.bootstrap_ci_mean(v.values)
        rows.append(dict(
            feature=f, n=int(v.notna().sum()),
            mean=v.mean(), median=v.median(), std=v.std(),
            p5=v.quantile(0.05), p95=v.quantile(0.95),
            ci95_lo=lo95, ci95_hi=hi95,
            env_lo=v.quantile(0.05), env_hi=v.quantile(0.95)))
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv"), index=False)

    # internal consistency: robust z per image across features
    X = base.set_index("image_id")[kept].astype(float)
    med = X.median()
    mad = (X - med).abs().median() * 1.4826
    z = (X - med) / mad.replace(0, np.nan)
    consistency = pd.DataFrame(dict(
        image_id=X.index,
        max_abs_robust_z=z.abs().max(axis=1).values,
        mean_abs_robust_z=z.abs().mean(axis=1).values))
    consistency["outlier"] = consistency.max_abs_robust_z > 3.5
    consistency.to_csv(os.path.join(
        C.TABLE_DIR, "baseline_consistency.csv"), index=False)

    # joint envelope: robust mean/cov (median + MAD-scaled IQR covariance)
    mu = X.median()
    sc = X.mad() if hasattr(X, "mad") else None
    scale = (X - mu).abs().median() * 1.4826
    Z = ((X - mu) / scale.replace(0, np.nan)).fillna(0)
    cov = np.cov(Z.values.T)
    env = dict(mu=mu.to_dict(), scale=scale.to_dict(), cov=cov.tolist(),
               features=kept)
    with open(os.path.join(C.TABLE_DIR, "envelope.json"), "w") as f:
        json.dump(env, f)
    C.log(f"baseline: {len(stats)} features, "
          f"outliers: {consistency[consistency.outlier].image_id.tolist()}")
    return stats, consistency, env


if __name__ == "__main__":
    import features as FT
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    rep = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
    kept, _ = FT.prune(F_, rep)
    run(F_, kept)
