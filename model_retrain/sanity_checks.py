"""sanity_checks.py - all 10 sanity checks, each returning pass/fail + numbers.

Labels = multi-Otsu masks; "accuracy" = agreement with that pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    ART_DIR,
    BATCHES,
    RNG_SEED,
    SEED,
    VAL_PX_PER_TILE,
    get_splits,
    leakage_assert,
    majority_baseline,
    per_class_metrics,
    tile_features,
)
from evaluate import eval_on_tiles, eval_test_images
from train import (
    build_xy,
    extract_subsample,
    load_tile,
    predict_labels,
    tile_index,
    tiles_for_images,
    train_rf,
)

CHECK_PX = 40_000   # pixel subsample per tile for check evaluations


def check_leakage(splits):
    """1. No parent image ID in more than one split (asserted)."""
    ok = leakage_assert(splits)
    return {"check": "1_leakage_assert", "pass": bool(ok),
            "detail": "all image IDs appear in exactly one split"}


def check_leakage_demo(splits, cache):
    """2. WRONG method on purpose: split tiles at random, ignoring the parent
    image -> inflated score, shows the size of the leakage effect."""
    tr_tiles = tiles_for_images(splits["train_val"]).tile_id.tolist()
    rng = np.random.RandomState(SEED)
    tiles = np.array(tr_tiles)
    rng.shuffle(tiles)
    n_tr = int(0.8 * len(tiles))
    tr, va = tiles[:n_tr], tiles[n_tr:]
    # count how many val tiles share a parent image with the train set
    idx = tile_index().set_index("tile_id")
    tr_imgs = set(idx.loc[tr, "image_id"])
    shared = sum(idx.loc[t, "image_id"] in tr_imgs for t in va)
    X, y = build_xy(list(tr), px_per_tile=30_000, cache=cache)
    clf = train_rf(X, y, seed=SEED)
    pooled, _ = eval_on_tiles(clf, list(va), cache=cache)
    detail = (f"WRONG split (random tiles): {shared}/{len(va)} val tiles "
              f"share a parent image with train; acc={pooled['accuracy']:.4f} "
              f"bal={pooled['balanced_accuracy']:.4f}")
    print(detail, flush=True)
    return {"check": "2_leakage_demonstration_WRONG", "pass": True,
            "detail": detail, "accuracy": pooled["accuracy"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "n_leaked_val_tiles": int(shared)}


def check_shuffled(splits, cache):
    """3. Train on randomly permuted labels -> must fall to ~majority level."""
    tr_ids = [i for j, f in enumerate(splits["folds"]) if j != 0 for i in f]
    va_ids = splits["folds"][0]
    tr_tiles = tiles_for_images(tr_ids).tile_id.tolist()
    va_tiles = tiles_for_images(va_ids).tile_id.tolist()
    X, y = build_xy(tr_tiles, px_per_tile=CHECK_PX, cache=cache)
    rng = np.random.RandomState(SEED + 42)
    y_shuf = rng.permutation(y)
    clf = train_rf(X, y_shuf, seed=SEED)
    pooled, _ = eval_on_tiles(clf, va_tiles, cache=cache)
    maj = np.mean(np.concatenate([y_shuf]) == np.bincount(y_shuf).argmax())
    acc = pooled["accuracy"]
    ok = acc < 0.90  # must not stay suspiciously high
    detail = (f"shuffled-label acc={acc:.4f} (majority fraction ~{maj:.3f}); "
              f"expected collapse to baseline")
    print(detail, flush=True)
    return {"check": "3_shuffled_labels", "pass": bool(ok),
            "detail": detail, "accuracy": acc, "majority_level": float(maj)}


def check_baselines(clf, splits):
    """4. Majority-class and plain multi-Otsu baselines on the test images."""
    idx = tile_index()
    rows = []
    from evaluate import raw_otsu_labels
    from common import list_locations, load_gray, segment_labels
    locs = {l["id"]: l for l in list_locations()}
    for iid in splits["test"]:
        lab = segment_labels(load_gray(locs[iid]["det"]["BSE"]))
        maj = majority_baseline(lab.ravel())
        rows.append({"image_id": iid, "majority_baseline": maj})
    # plain per-TILE multi-Otsu (no morphology) vs labels, on test tiles
    otsu_rows = []
    for tid in tiles_for_images(splits["test"]).tile_id:
        t = load_tile(tid)
        raw = raw_otsu_labels(t["bse"].astype(np.float32))
        acc = (raw == t["labels"]).mean()
        otsu_rows.append({"tile_id": tid, "raw_otsu_agreement": float(acc)})
    bdf = pd.DataFrame(rows)
    odf = pd.DataFrame(otsu_rows)
    detail = (f"majority-class acc mean={bdf.majority_baseline.mean():.4f}; "
              f"raw per-tile multi-Otsu agreement mean="
              f"{odf.raw_otsu_agreement.mean():.4f}")
    print(detail, flush=True)
    return {"check": "4_baselines", "pass": True, "detail": detail,
            "majority_mean": float(bdf.majority_baseline.mean()),
            "raw_otsu_mean": float(odf.raw_otsu_agreement.mean())}, bdf, odf


def check_tile_edge(clf, splits, n_images=3):
    """6+7. Predict per-tile and as whole image on test images:
    edge accuracy vs interior; stitched tiles vs whole-image prediction."""
    from common import list_locations, load_gray
    locs = {l["id"]: l for l in list_locations()}
    rows = []
    edge = 10
    for iid in splits["test"][:n_images]:
        loc = locs[iid]
        imgs = {d: load_gray(loc["det"][d]) for d in ("BSE", "Inlens", "ETD")}
        feat_w = tile_features(imgs["BSE"], imgs["Inlens"], imgs["ETD"])
        pred_w = predict_labels(clf, feat_w)
        del feat_w
        lab_w = np.load(ART_DIR / f"pred_{iid}.npz")["labels"]
        h, w = lab_w.shape
        h2, w2 = h // 2, w // 2
        quads = [(slice(0, h2), slice(0, w2)), (slice(0, h2), slice(w2, w)),
                 (slice(h2, h), slice(0, w2)), (slice(h2, h), slice(w2, w))]
        # per-tile predictions
        diffs = []
        edge_hits, edge_tot, int_hits, int_tot = 0, 0, 0, 0
        for q, (rs, cs) in enumerate(quads):
            t = load_tile(f"{iid}_q{q}")
            feat = tile_features(t["bse"], t["inl"], t["etd"])
            pred_t = predict_labels(clf, feat)
            diffs.append(pred_t)
            lab_t = lab_w[rs, cs]
            th, tw = lab_t.shape
            border = np.zeros_like(lab_t, bool)
            border[:edge] = border[-edge:] = True
            border[:, :edge] = border[:, -edge:] = True
            ok = pred_t == lab_t
            edge_hits += ok[border].sum(); edge_tot += border.sum()
            int_hits += ok[~border].sum(); int_tot += (~border).sum()
        stitched = np.empty_like(pred_w)
        for q, (rs, cs) in enumerate(quads):
            stitched[rs, cs] = diffs[q]
        mismatch = float((stitched != pred_w).mean())
        rows.append({
            "image_id": iid,
            "edge_acc": float(edge_hits / edge_tot),
            "interior_acc": float(int_hits / int_tot),
            "stitch_vs_whole_mismatch": mismatch,
        })
        print(f"{iid}: edge={rows[-1]['edge_acc']:.4f} "
              f"interior={rows[-1]['interior_acc']:.4f} "
              f"stitch-mismatch={mismatch:.4f}", flush=True)
    df = pd.DataFrame(rows)
    ok6 = (df.interior_acc - df.edge_acc).mean() < 0.05
    ok7 = df.stitch_vs_whole_mismatch.mean() < 0.02
    return [
        {"check": "6_tile_edge", "pass": bool(ok6),
         "detail": f"edge acc {df.edge_acc.mean():.4f} vs interior "
                   f"{df.interior_acc.mean():.4f} (mean gap "
                   f"{(df.interior_acc - df.edge_acc).mean():.4f})",
         "edge_acc": float(df.edge_acc.mean()),
         "interior_acc": float(df.interior_acc.mean())},
        {"check": "7_reassembly", "pass": bool(ok7),
         "detail": f"stitched-vs-whole mismatch mean "
                   f"{df.stitch_vs_whole_mismatch.mean():.4f}",
         "mismatch_mean": float(df.stitch_vs_whole_mismatch.mean())},
        df,
    ]


def check_reproducibility(splits, cache):
    """8. Same seed twice -> identical predictions; 3 seeds -> spread."""
    tr_ids = [i for j, f in enumerate(splits["folds"]) if j != 0 for i in f]
    va_ids = splits["folds"][0]
    tr_tiles = tiles_for_images(tr_ids).tile_id.tolist()
    va_tiles = tiles_for_images(va_ids).tile_id.tolist()
    X, y = build_xy(tr_tiles, px_per_tile=CHECK_PX, cache=cache)
    clf_a = train_rf(X, y, seed=SEED)
    clf_b = train_rf(X, y, seed=SEED)
    Xv, yv = build_xy(va_tiles[:4], px_per_tile=20_000, augment=False,
                      seed=RNG_SEED + 999, cache=cache)
    identical = bool((clf_a.predict(Xv) == clf_b.predict(Xv)).all())
    accs = []
    for s in (SEED, SEED + 1, SEED + 2):
        clf_s = train_rf(X, y, seed=s)
        accs.append(float((clf_s.predict(Xv) == yv).mean()))
    detail = (f"same-seed predictions identical: {identical}; "
              f"3-seed acc spread: {[round(a, 4) for a in accs]}")
    print(detail, flush=True)
    return {"check": "8_reproducibility", "pass": bool(identical),
            "detail": detail, "seed_accs": accs,
            "seed_spread": float(max(accs) - min(accs))}


def check_heldout_batch(splits, cache):
    """9. Train on two batches, test on the third (all its images)."""
    idx = tile_index()
    rows = []
    for held in BATCHES:
        tr_ids = idx[(idx.batch != held)].image_id.unique().tolist()
        te_ids = idx[(idx.batch == held)].image_id.unique().tolist()
        tr_tiles = idx[idx.image_id.isin(tr_ids)].tile_id.tolist()
        te_tiles = idx[idx.image_id.isin(te_ids)].tile_id.tolist()
        X, y = build_xy(tr_tiles, px_per_tile=CHECK_PX, cache=cache)
        clf = train_rf(X, y, seed=SEED)
        pooled, _ = eval_on_tiles(clf, te_tiles, px_per_tile=20_000,
                                  cache=cache)
        rows.append({"held_out_batch": held,
                     "n_train_images": len(tr_ids),
                     "n_test_images": len(te_ids),
                     "accuracy": pooled["accuracy"],
                     "balanced_accuracy": pooled["balanced_accuracy"]})
        print(f"held-out {held}: acc={pooled['accuracy']:.4f} "
              f"bal={pooled['balanced_accuracy']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    ok = bool((df.accuracy > 0.85).all())  # should still beat majority
    return {"check": "9_heldout_batch", "pass": ok,
            "detail": "; ".join(
                f"{r.held_out_batch}: {r.accuracy:.4f}"
                for r in df.itertuples())}, df


def test_tile_scores(splits):
    """Per-tile accuracy on test images, derived from stored whole-image
    predictions (quadrants == the four tiles)."""
    idx = tile_index()
    rows = []
    for iid in splits["test"]:
        z = np.load(ART_DIR / f"pred_{iid}.npz")
        lab, pred = z["labels"], z["pred"]
        h, w = lab.shape
        h2, w2 = h // 2, w // 2
        quads = [(slice(0, h2), slice(0, w2)), (slice(0, h2), slice(w2, w)),
                 (slice(h2, h), slice(0, w2)), (slice(h2, h), slice(w2, w))]
        for q, (rs, cs) in enumerate(quads):
            lt, pt = lab[rs, cs], pred[rs, cs]
            m = per_class_metrics(lt.ravel(), pt.ravel())
            rows.append({"tile_id": f"{iid}_q{q}", "image_id": iid,
                         "batch": loc_batch(idx, iid),
                         "accuracy": m["accuracy"],
                         "balanced_accuracy": m["balanced_accuracy"]})
    return pd.DataFrame(rows)


def loc_batch(idx, iid):
    return idx[idx.image_id == iid].batch.iloc[0]
