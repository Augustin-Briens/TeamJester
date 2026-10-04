"""Regenerate EVERYTHING from the raw images.

    python3 run_all.py            # full pipeline
    python3 run_all.py --fast     # skip DFN

Stages: segmentation+validation -> features -> baseline -> deltas ->
categorisation validation -> DFN -> figures/workbooks/PDFs/NOTES.
Intermediate tables are cached per stage so a rerun only redoes what is
missing; pass --clean to force a full rebuild.
"""
import os
import sys
import time
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def need(*paths):
    return not all(os.path.exists(p) for p in paths)


def stage_segmentation():
    if not need(os.path.join(C.TABLE_DIR, "segment_info.csv")):
        return
    import segment as S
    S.main()


def stage_features():
    if not need(os.path.join(C.TABLE_DIR, "features.csv")):
        return
    import features as FT
    FT.run()


def stage_stats():
    import pandas as pd
    import features as FT
    import baseline as B
    import deltas as DL
    import categorise as CAT
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    rep = FT.repeatability(F_)
    kept, _ = FT.prune(F_, rep)
    q = pd.read_csv(os.path.join(C.TABLE_DIR, "quality_guards.csv"))
    excl = tuple(q[q.flagged].image_id)

    B.run(F_, kept, ())                     # baseline_stats.csv + envelope
    stats, cons, env = B.run(F_, kept, excl)
    os.rename(os.path.join(C.TABLE_DIR, "baseline_stats.csv"),
              os.path.join(C.TABLE_DIR, "baseline_stats_noflag.csv"))
    os.rename(os.path.join(C.TABLE_DIR, "baseline_consistency.csv"),
              os.path.join(C.TABLE_DIR, "baseline_consistency_noflag.csv"))
    # restore all-images baseline outputs
    B.run(F_, kept, ())

    DL.run(F_, kept, ())                   # deltas.csv (kept features)
    D = DL.run(F_, [f for f in pd.read_csv(os.path.join(
        C.TABLE_DIR, "feature_meta.csv")).feature], ())  # all features
    Dk = D[D.feature.isin(kept)]
    Dk.to_csv(os.path.join(C.TABLE_DIR, "deltas.csv"), index=False)
    D[~D.feature.isin(kept)].to_csv(
        os.path.join(C.TABLE_DIR, "deltas_dropped.csv"), index=False)
    DL.run(F_, kept, excl)
    os.rename(os.path.join(C.TABLE_DIR, "deltas.csv"),
              os.path.join(C.TABLE_DIR, "deltas_noflag.csv"))
    DL.run(F_, kept, ())                   # restore deltas.csv

    # categorisation validation (LOO + controls + group-size curve)
    full = F_[F_.subset == "full"]
    safe = [f for f in kept if f not in CAT.ARTEFACT_RISK]
    P1, S1 = CAT.loo_validate(full, kept, rep, exclude_ids=excl,
                              tag="allfeatures")
    P1.to_csv(os.path.join(C.TABLE_DIR, "loo_predictions.csv"), index=False)
    P2, _ = CAT.loo_validate(full, safe, rep, exclude_ids=excl, tag="safe")
    P3, _ = CAT.loo_validate(full, safe, rep, exclude_ids=excl,
                             tag="safe_noflag")
    CAT.images_needed(full, kept, rep, exclude_ids=excl).to_csv(
        os.path.join(C.TABLE_DIR, "images_needed.csv"), index=False)
    CAT.images_needed(full, safe, rep, exclude_ids=excl).to_csv(
        os.path.join(C.TABLE_DIR, "images_needed_safe.csv"), index=False)
    rows = []
    for name, fn in (("all-features", "loo_predictions_allfeatures.csv"),
                     ("artefact-safe", "loo_predictions_safe.csv"),
                     ("artefact-safe-no-flag",
                      "loo_predictions_safe_noflag.csv")):
        P = pd.read_csv(os.path.join(C.TABLE_DIR, fn))
        acc = P.groupby("batch").apply(
            lambda g: (g.batch == g.pred_batch).mean())
        rows.append(dict(
            variant=name, overall_acc=(P.batch == P.pred_batch).mean(),
            B1_acc=acc.get("Batch_1"), B2_acc=acc.get("Batch_2"),
            B3_acc=acc.get("Batch_3"),
            false_alarm=((P.batch == "Batch_3") &
                         (P.call_inout == "OUT")).mean(), n=len(P)))
    rows.append(dict(variant="shuffled-label control (mean of 20)",
                     overall_acc=S1.get("shuffled_accuracy_mean"), n=20))
    pd.DataFrame(rows).to_csv(
        os.path.join(C.TABLE_DIR, "loo_summary.csv"), index=False)


def stage_dfn():
    if need(os.path.join(C.TABLE_DIR, "dfn_capacity.csv")):
        return
    import pandas as pd
    import dfn
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    dfn.run(F_)


def stage_reports():
    import pandas as pd
    import common as C
    import reports as R
    locs = C.find_locations()
    t = C.TABLE_DIR
    F_ = pd.read_csv(os.path.join(t, "features.csv"))
    stats = pd.read_csv(os.path.join(t, "baseline_stats.csv"))
    kept = stats.feature.tolist()
    D = pd.read_csv(os.path.join(t, "deltas.csv"))
    P = pd.read_csv(os.path.join(t, "loo_predictions.csv"))
    import categorise as CAT
    rep = pd.read_csv(os.path.join(t, "repeatability.csv"))
    feats = CAT.select_features(F_[F_.subset == "full"], kept, rep)
    R.fig_feature_panels(F_, kept, stats)
    R.fig_ranked_deltas(D)
    R.fig_distributions(locs)
    R.fig_depth_profiles(locs)
    R.fig_dfn()
    R.fig_categorisation_map(P, F_[F_.subset == "full"], feats)
    for _, r in D[(D.classification.isin(["different", "suggestive"])) &
                  (D.feature.isin(["pore_frac", "large_gap_frac",
                                   "silicon_frac", "gr_st_coherence",
                                   "bse_bulk_texture"]))].iterrows():
        R.fig_evidence(r, "pore" if "pore" in r.feature or
                       "large" in r.feature else "silicon", locs)
    R.build_workbooks(locs)
    R.build_all_pdfs(locs)
    R.write_notes(locs)


def main():
    fast = "--fast" in sys.argv
    clean = "--clean" in sys.argv
    if clean:
        import shutil
        for d in (C.OUT, C.MASK_DIR):
            if os.path.isdir(d):
                shutil.rmtree(d)
    t0 = time.time()
    stage_segmentation(); C.log(f"seg done {time.time()-t0:.0f}s")
    stage_features(); C.log(f"feat done {time.time()-t0:.0f}s")
    stage_stats(); C.log(f"stats done {time.time()-t0:.0f}s")
    if not fast:
        stage_dfn(); C.log(f"dfn done {time.time()-t0:.0f}s")
    stage_reports(); C.log(f"reports done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
