"""evaluate.py - grouped CV, final test evaluation, metrics, figures, xlsx.

Accuracy here measures AGREEMENT WITH the multi-Otsu label pipeline
(qc_real.py) - the labels were produced by a threshold, not a person.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import filters, morphology

from common import (
    ART_DIR,
    BATCHES,
    CLASS_IDS,
    CLASS_NAMES,
    FIG_DIR,
    MIN_BRIGHT_PX,
    OUTLIER_IDS,
    RNG_SEED,
    SEED,
    SIGMA,
    TILES_DIR,
    bootstrap_image_ci,
    get_splits,
    leakage_assert,
    list_locations,
    load_gray,
    majority_baseline,
    normalize,
    per_class_metrics,
    segment_labels,
    tile_features,
    FEATURE_NAMES,
)
from train import (
    build_xy,
    extract_subsample,
    load_tile,
    predict_labels,
    predict_proba_labels,
    save_model,
    tile_index,
    tiles_for_images,
    train_rf,
)

VAL_PX = 50_000
TEST_CHUNK = 256

BATCH_COLORS = {"Batch_1": "#4477AA", "Batch_2": "#EE6677", "Batch_3": "#228833"}
CLASS_COLORS = {"pore": "#35A7FF", "bulk": "#8C8C8C", "bright": "#FF7A00"}


# ----------------------------------------------------------------- helpers
def eval_on_tiles(clf, tile_ids, px_per_tile=VAL_PX, seed=RNG_SEED,
                  cache=None):
    """Pooled + per-tile metrics on a deterministic pixel subsample."""
    rows, yt_all, yp_all = [], [], []
    for tid in tile_ids:
        X, y = build_xy([tid], px_per_tile=px_per_tile, augment=False,
                        seed=seed + 999, cache=cache)
        yp = clf.predict(X)
        m = per_class_metrics(y, yp)
        rows.append({"tile_id": tid, "accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"]})
        yt_all.append(y)
        yp_all.append(yp)
    pooled = per_class_metrics(np.concatenate(yt_all), np.concatenate(yp_all))
    return pooled, pd.DataFrame(rows)


def raw_otsu_labels(bse_gray):
    """Plain 3-class multi-Otsu WITHOUT morphology cleanup (baseline)."""
    s = filters.gaussian(normalize(bse_gray), sigma=SIGMA)
    t_dark, t_bright = filters.threshold_multiotsu(s, classes=3)
    lab = np.ones(s.shape, dtype=np.uint8)
    lab[s < t_dark] = 0
    lab[s > t_bright] = 2
    return lab


def phase_fracs(labels):
    out = {}
    n = labels.size
    for c, name in zip(CLASS_IDS, CLASS_NAMES):
        out[name] = float((labels == c).sum() / n)
    return out


# ----------------------------------------------------------------- stages
def run_cv(splits, cache):
    """5-fold grouped CV. Returns fold dataframe + pooled metric dicts."""
    idx = tile_index()
    rows = []
    pooled_by_fold = {}
    for k, val_ids in enumerate(splits["folds"]):
        tr_ids = [i for j, f in enumerate(splits["folds"]) if j != k
                  for i in f]
        tr_tiles = tiles_for_images(tr_ids).tile_id.tolist()
        va_tiles = tiles_for_images(val_ids).tile_id.tolist()
        t0 = time.time()
        X, y = build_xy(tr_tiles, cache=cache)
        clf = train_rf(X, y)
        del X, y
        pooled, per_tile = eval_on_tiles(clf, va_tiles, cache=cache)
        # train-side accuracy on a fresh train-pixel subsample (overfit check)
        tr_pooled, _ = eval_on_tiles(
            clf, tr_tiles[:20], px_per_tile=5_000, seed=RNG_SEED + 555)
        rows.append({
            "fold": k, "n_val_images": len(val_ids),
            "val_images": ",".join(sorted(val_ids)),
            "accuracy": pooled["accuracy"],
            "balanced_accuracy": pooled["balanced_accuracy"],
            "train_accuracy_subsample": tr_pooled["accuracy"],
            "seconds": round(time.time() - t0, 1),
        })
        pooled_by_fold[k] = pooled
        print(f"fold {k}: acc={pooled['accuracy']:.4f} "
              f"bal={pooled['balanced_accuracy']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    return df, pooled_by_fold


def train_final(splits, cache):
    """Train on ALL train_val tiles; save the model + metadata."""
    tr_tiles = tiles_for_images(splits["train_val"]).tile_id.tolist()
    t0 = time.time()
    X, y = build_xy(tr_tiles, cache=cache)
    print(f"final fit: {X.shape[0]:,} px x {X.shape[1]} features "
          f"from {len(tr_tiles)} tiles")
    clf = train_rf(X, y)
    save_model(clf, ART_DIR / "rf_model.joblib", {
        "features": list(FEATURE_NAMES), "rf": {k: str(v) for k, v in
        clf.get_params().items() if k in
        ("n_estimators", "max_depth", "min_samples_leaf", "max_features")},
        "train_images": splits["train_val"], "seed": SEED,
        "labels": "3-class multi-Otsu masks (qc_real.py); agreement != truth",
    })
    print(f"final model trained in {time.time() - t0:.0f}s")
    return clf


def eval_test_images(clf, splits):
    """Full-pixel prediction on every held-out test image."""
    locs = {l["id"]: l for l in list_locations()}
    rows, store = [], {}
    for iid in splits["test"]:
        loc = locs[iid]
        imgs = {d: load_gray(loc["det"][d]) for d in ("BSE", "Inlens", "ETD")}
        labels = segment_labels(imgs["BSE"])
        feat = tile_features(imgs["BSE"], imgs["Inlens"], imgs["ETD"])
        pred, conf = predict_proba_labels(clf, feat)
        m = per_class_metrics(labels.ravel(), pred.ravel())
        lf, pf = phase_fracs(labels), phase_fracs(pred)
        mae = float(np.mean([abs(lf[n] - pf[n]) for n in CLASS_NAMES]))
        np.savez_compressed(
            ART_DIR / f"pred_{iid}.npz",
            labels=labels, pred=pred, conf=conf.astype(np.float16),
        )
        rows.append({
            "image_id": iid, "batch": loc["batch"],
            "outlier": iid in OUTLIER_IDS,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "frac_mae": mae,
            **{f"frac_{n}_label": lf[n] for n in CLASS_NAMES},
            **{f"frac_{n}_pred": pf[n] for n in CLASS_NAMES},
        })
        store[iid] = m
        print(f"test {iid}: acc={m['accuracy']:.4f} "
              f"bal={m['balanced_accuracy']:.4f} fracMAE={mae:.4f}",
              flush=True)
    return pd.DataFrame(rows), store


# ----------------------------------------------------------------- figures
def fig_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("label (multi-Otsu)")
    ax.set_title("Confusion matrix, row-normalized")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_per_class(per_class, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(CLASS_NAMES))
    w = 0.2
    for j, m in enumerate(("precision", "recall", "f1", "iou")):
        vals = [per_class[n][m] for n in CLASS_NAMES]
        ax.bar(x + (j - 1.5) * w, vals, w, label=m)
        for xi, v in zip(x + (j - 1.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, CLASS_NAMES); ax.set_ylim(0, 1.15)
    ax.set_ylabel("score"); ax.legend(ncol=4, fontsize=8)
    ax.set_title("Per-class metrics on held-out test images (pixel-level)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_train_val(fold_df, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(fold_df))
    ax.bar(x - 0.2, fold_df.train_accuracy_subsample, 0.4,
           label="train px (subsample)")
    ax.bar(x + 0.2, fold_df.accuracy, 0.4, label="validation tiles")
    ax.set_xticks(x, [f"fold {k}" for k in fold_df.fold])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("accuracy"); ax.legend()
    ax.set_title("Train vs validation accuracy (overfitting check)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_confidence(store_dir, image_ids, path):
    """Accuracy vs confidence bins pooled over test images."""
    bins = np.linspace(0.33, 1.0, 12)
    corrs, tots = np.zeros(len(bins)), np.zeros(len(bins))
    for iid in image_ids:
        z = np.load(store_dir / f"pred_{iid}.npz")
        lab, pred, conf = z["labels"].ravel(), z["pred"].ravel(), z["conf"].astype(np.float32).ravel()
        ind = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 1)
        tots += np.bincount(ind, minlength=len(bins))
        corrs += np.bincount(ind, weights=(lab == pred), minlength=len(bins))
    acc = np.divide(corrs, tots, out=np.full(len(bins), np.nan), where=tots > 0)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(range(len(bins)), acc, tick_label=[f"{b:.2f}" for b in bins])
    ax.set_ylim(0, 1.0); ax.set_xlabel("confidence (max class prob)")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Is it right when it is sure? (held-out test pixels)")
    for i, (a, t) in enumerate(zip(acc, tots)):
        if t > 0:
            ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return pd.DataFrame({"conf_bin_lo": bins, "accuracy": acc,
                         "n_pixels": tots.astype(int)})


def fig_side_by_side(image_ids, locs, path):
    """For 6 test images: BSE | label | prediction | disagreement."""
    fig, axes = plt.subplots(len(image_ids), 4,
                             figsize=(13, 2.2 * len(image_ids)))
    cmap = matplotlib.colors.ListedColormap(
        [CLASS_COLORS[n] for n in CLASS_NAMES])
    for r, iid in enumerate(image_ids):
        z = np.load(ART_DIR / f"pred_{iid}.npz")
        lab, pred = z["labels"], z["pred"]
        bse = normalize(load_gray(locs[iid]["det"]["BSE"]))
        ds = np.s_[::2, ::2]
        axes[r, 0].imshow(bse[ds], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(lab[ds], cmap=cmap, vmin=0, vmax=2)
        axes[r, 2].imshow(pred[ds], cmap=cmap, vmin=0, vmax=2)
        diff = (lab != pred).astype(float)
        axes[r, 3].imshow(diff[ds], cmap="hot", vmin=0, vmax=1)
        for c, tt in enumerate(("BSE", "label (Otsu)", "RF prediction",
                                "disagreement")):
            axes[r, c].set_title(f"{iid} - {tt}", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_worst_tiles(worst, clf, path):
    """Grid of the worst-scoring tiles with label/pred/disagreement."""
    n = len(worst)
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.0 * n))
    cmap = matplotlib.colors.ListedColormap(
        [CLASS_COLORS[nm] for nm in CLASS_NAMES])
    for r, (_, row) in enumerate(worst.iterrows()):
        t = load_tile(row.tile_id)
        feat = tile_features(t["bse"], t["inl"], t["etd"])
        pred = predict_labels(clf, feat)
        lab = t["labels"]
        ds = np.s_[::2, ::2]
        axes[r, 0].imshow(lab[ds], cmap=cmap, vmin=0, vmax=2)
        axes[r, 1].imshow(pred[ds], cmap=cmap, vmin=0, vmax=2)
        axes[r, 2].imshow((lab != pred)[ds].astype(float), cmap="hot",
                          vmin=0, vmax=1)
        for c, tt in enumerate(("label", "pred", "disagree")):
            axes[r, c].set_title(
                f"{row.tile_id} acc={row.accuracy:.3f} - {tt}", fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_importances(clf, path):
    imp = pd.Series(clf.feature_importances_, index=FEATURE_NAMES)
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(range(len(imp)), imp.values, tick_label=imp.index)
    ax.set_xlabel("feature importance (mean decrease in impurity)")
    ax.set_title("Which features drive the classifier")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return imp
