"""One-command clean report — every number generated from saved CSVs.

Reads:
  validated_comparison/per_image_features.csv  (per-image markers)
  image_acquisition.csv                        (acquisition groups)
  validated_comparison/objects_classified.csv  (object counts)
  dfn_output/dfn_paired_differences.csv        (model indicators)
Writes:
  clean_report.md          — the plain report
  report_numbers.json      — every number, for the check script
Usage: .venv/bin/python build_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

RNG = np.random.default_rng(0)
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
ALPHA = 0.05
N_PERM = 5000

# Declared families (decided before looking at results):
PRIMARY_FRACTIONS = ["pore_frac", "si_candidate_frac",
                     "uncertain_bright_frac", "bright_frac_raw"]
BOUNDARY_MARKERS = ["si_d50_um", "si_d90_um", "si_clustering_R",
                    "pore_d50_um", "pore_anisotropy",
                    "pore_chord_h_um", "pore_chord_v_um"]
COUNT_MARKERS = ["pores_per_mpx", "pores_per_mpx_noise_adj",
                 "si_n_particles"]
ACQ_FEATURES = ["noise_mad", "sharpness_lapvar", "img_p1",
                "img_p50", "img_p99", "si_bulk_contrast",
                "shading_slope"]
DECLARED_FAMILY = PRIMARY_FRACTIONS + BOUNDARY_MARKERS + COUNT_MARKERS


def exact_perm_median(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exact/Monte-Carlo permutation p on the median difference."""
    d = float(np.median(x) - np.median(y))
    pool = np.concatenate([x, y])
    n = len(x)
    # exact if small enough
    if len(pool) <= 20:
        from itertools import combinations as comb
        tot, ge = 0, 0
        for idx in comb(range(len(pool)), n):
            xa = pool[list(idx)]
            ya = pool[[i for i in range(len(pool)) if i not in idx]]
            tot += 1
            ge += abs(np.median(xa) - np.median(ya)) >= abs(d) - 1e-12
        return d, (ge + 1) / (tot + 1)
    ge = 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        ge += abs(np.median(pool[:n]) - np.median(pool[n:])) >= abs(d) - 1e-12
    return d, (ge + 1) / (N_PERM + 1)


def mdd(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Minimum detectable |median difference| at this n via the
    permutation null distribution."""
    pool = np.concatenate([x, y])
    n = len(x)
    ds = []
    for _ in range(3000):
        RNG.shuffle(pool)
        ds.append(np.median(pool[:n]) - np.median(pool[n:]))
    return float(np.quantile(np.abs(ds), 1 - alpha))


def bh(pvals: list[float], q: float = 0.10) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.clip(adj, 0, 1)
    return out


def stratified_perm(df: pd.DataFrame, col: str, b1: str, b2: str,
                    groups: list) -> tuple[float, float, int]:
    """Median difference of group-centred values; batch labels permuted
    within groups only. Returns (diff, p, n_pairs)."""
    d = df[df.group_id.isin(groups)][["batch", "group_id", col]].dropna()
    d = d[d.batch.isin([b1, b2])].copy()
    d["gc"] = d[col] - d.groupby("group_id")[col].transform("median")
    pairs = d.groupby(["group_id", "batch"]).size().unstack(fill_value=0)
    usable = pairs[(pairs[b1] > 0) & (pairs[b2] > 0)]
    if not len(usable):
        return np.nan, np.nan, 0
    obs = float(d[d.batch == b1].gc.median() - d[d.batch == b2].gc.median())
    ge = 0
    for _ in range(N_PERM):
        dd = d.copy()
        for _, idx in dd.groupby("group_id").groups.items():
            if len(dd.loc[idx]) > 1:
                dd.loc[idx, "batch"] = RNG.permutation(dd.loc[idx, "batch"])
        v1, v2 = dd[dd.batch == b1].gc, dd[dd.batch == b2].gc
        if len(v1) and len(v2):
            ge += abs(v1.median() - v2.median()) >= abs(obs) - 1e-12
    return obs, (ge + 1) / (N_PERM + 1), int(usable.shape[0])


def main() -> None:
    f = pd.read_csv("validated_comparison/per_image_features.csv")
    acq = pd.read_csv("image_acquisition.csv")
    objs = pd.read_csv("validated_comparison/objects_classified.csv")
    f = f[f.batch != "New_Images_Batch"]
    acq["group_id"] = acq.group_id.astype(str)
    m = f.merge(acq[["image_id", "group_id"]], on="image_id", how="left")

    out: dict = {"n_images": int(len(f)),
                 "per_batch_n": f.batch.value_counts().to_dict()}

    # ---- 1. primary fractions -------------------------------------------
    frac = {}
    for c in PRIMARY_FRACTIONS:
        frac[c] = {b: {"med": float(f[f.batch == b][c].median()),
                       "lo": float(f[f.batch == b][c].min()),
                       "hi": float(f[f.batch == b][c].max())}
                   for b in BATCHES}
    out["fractions"] = frac

    # ---- 2. acquisition partition + chance levels ------------------------
    n_g, n_i, n_b = acq[acq.batch != "New_Images_Batch"].group_id.nunique(), \
        len(m), 3
    out["chance"] = {"after_group": round(1 - (n_g - 1) / (n_i - 1), 3),
                     "after_batch": round(1 - (n_b - 1) / (n_i - 1), 3)}
    part = []
    for c in DECLARED_FAMILY + ACQ_FEATURES:
        if c not in m.columns:
            continue
        s = pd.to_numeric(m[c], errors="coerce")
        d = m.assign(v=s).dropna(subset=["v"])
        if d.v.var() == 0:
            continue
        rg = ((d.v - d.groupby("group_id").v.transform("mean")) ** 2
              ).mean() / d.v.var()
        rb = ((d.v - d.groupby("batch").v.transform("mean")) ** 2
              ).mean() / d.v.var()
        part.append(dict(marker=c, resid_group=round(float(rg), 3),
                         resid_batch=round(float(rb), 3)))
    out["partition"] = part

    # ---- 3. unblocked comparison on the declared family -----------------
    rows = []
    for c in DECLARED_FAMILY:
        for b1, b2 in combinations(BATCHES, 2):
            x = pd.to_numeric(f[f.batch == b1][c],
                              errors="coerce").dropna().values
            y = pd.to_numeric(f[f.batch == b2][c],
                              errors="coerce").dropna().values
            d, p = exact_perm_median(x, y)
            rows.append(dict(marker=c, pair=f"{b1}-{b2}", diff=round(d, 4),
                             p=round(p, 5),
                             mdd=round(mdd(x, y), 4)))
    ub = pd.DataFrame(rows)
    ub["padj"] = bh(ub.p.values, 0.10).round(4)
    out["unblocked"] = ub.to_dict("records")
    out["unblocked_nominal"] = int((ub.p < ALPHA).sum())
    out["unblocked_expected_chance"] = round(len(ub) * ALPHA, 1)
    out["unblocked_bh_survivors"] = int((ub.padj < 0.10).sum())

    # ---- 4. blocked comparison (stratified permutation) ------------------
    mixed = [g for g, s in acq[acq.batch != "New_Images_Batch"]
             .groupby("group_id").batch.unique().items() if len(s) > 1]
    out["mixed_groups"] = sorted(mixed)
    shared = {}
    for b1, b2 in combinations(BATCHES, 2):
        shared[f"{b1}-{b2}"] = sorted(
            g for g in mixed
            if {b1, b2} <= set(acq[acq.group_id == g].batch))
    out["shared_sessions"] = shared
    brows = []
    for c in DECLARED_FAMILY:
        for b1, b2 in combinations(BATCHES, 2):
            d, p, npg = stratified_perm(m, c, b1, b2, mixed)
            brows.append(dict(marker=c, pair=f"{b1}-{b2}",
                              diff=None if np.isnan(d) else round(d, 4),
                              p=None if np.isnan(p) else round(p, 4),
                              n_pair_groups=npg))
    out["blocked"] = brows
    out["blocked_nominal"] = int(sum(
        1 for r in brows if r["p"] is not None and r["p"] < ALPHA))

    # ---- 5. objects + problem photos -------------------------------------
    out["objects"] = {
        "total": int(len(objs)),
        "si_particle": int((objs.kind == "si_particle").sum()),
        "bright_fine": int((objs.kind == "bright_fine").sum())}
    prob = f[f.image_id.isin(["img_4ih2ggld", "img_5n1q8atc"])]
    out["problem_photos"] = prob[
        ["image_id", "bright_frac_raw", "si_candidate_frac",
         "uncertain_bright_frac"]].round(4).to_dict("records")

    # ---- 6. DFN (appendix only) ------------------------------------------
    try:
        dfn = pd.read_csv("dfn_output/dfn_paired_differences.csv")
        st = dfn[dfn.indicator == "si_stress_MPa"][
            ["pair", "median_paired_diff", "verdict"]]
        out["dfn_stress"] = st.to_dict("records")
    except Exception:
        out["dfn_stress"] = "dfn_paired_differences.csv unavailable"

    # ---- 7. B1 bright fields vs B3 range ---------------------------------
    b3max = f[f.batch == "Batch_3"].bright_frac_raw.max()
    b1 = f[f.batch == "Batch_1"][["image_id", "bright_frac_raw"]]
    out["bright_b1_vs_b3max"] = dict(
        b3_max=round(float(b3max), 4),
        b1_above=b1[b1.bright_frac_raw > b3max].image_id.tolist(),
        b1_above_vals=b1[b1.bright_frac_raw > b3max]
        .bright_frac_raw.round(4).tolist(),
        b1_ge_b3hi=b1[b1.bright_frac_raw >=
                      f[f.batch == "Batch_3"].bright_frac_raw.quantile(0.75)]
        .image_id.tolist())

    with open("report_numbers.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("partition", "unblocked", "blocked",
                                   "fractions")}, indent=2, default=str))
    print("wrote report_numbers.json")


if __name__ == "__main__":
    main()
