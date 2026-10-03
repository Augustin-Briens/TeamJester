"""Shared constants, IO, label generation and features for model_retrain.

Labels are produced by the SAME 3-class multi-Otsu segmentation as qc_real.py
(the team's existing rule-based pipeline). Accuracy of the learned model
therefore measures AGREEMENT WITH multi-Otsu + morphology cleanup, not
correctness against a human-labelled ground truth. This is stated
prominently in every output.

Assumption (carried from earlier tasks): bright BSE particles are ASSUMED
silicon-based and the grey bulk is ASSUMED graphite, from image appearance
only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tifffile
from skimage import filters, morphology

# ------------------------------------------------------------------ settings
SEED = 20241003
RNG_SEED = SEED
SIGMA = 1.5                 # smoothing before thresholding (qc_real.py)
MIN_BRIGHT_PX = 150         # bright specks below this are dropped (qc_real.py)
N_TEST_IMAGES_PER_BATCH = {"Batch_1": 2, "Batch_2": 2, "Batch_3": 3}
N_FOLDS = 5
TRAIN_PX_PER_TILE = 30_000          # random pixel subsample for fitting
VAL_PX_PER_TILE = 50_000            # random pixel subsample for fold metrics
DETECTORS = ("BSE", "Inlens", "ETD")  # ETD falls back to SE when absent
CLASS_NAMES = ("pore", "bulk", "bright")
CLASS_IDS = (0, 1, 2)

DATA_ROOT = Path(__file__).resolve().parent.parent / "Hackathon-Polaron"
OUT_DIR = Path(__file__).resolve().parent
TILES_DIR = OUT_DIR / "tiles"
ART_DIR = OUT_DIR / "artifacts"
FIG_DIR = OUT_DIR / "figures"

BATCHES = ("Batch_1", "Batch_2", "Batch_3")
# outlier images from earlier tasks: kept in the data (per spec, tiles of an
# image stay together) but flagged everywhere they appear.
OUTLIER_IDS = {"4ih2ggld", "5n1q8atc"}


# ------------------------------------------------------------------ data IO
def list_locations():
    """Return list of dicts: id, batch, paths per detector."""
    locs = []
    for batch in BATCHES:
        bdir = DATA_ROOT / batch
        for f in sorted(bdir.glob("img_*_BSE.tif")):
            iid = f.name[len("img_"):-len("_BSE.tif")]
            det = {"BSE": f}
            for d in ("Inlens", "ETD", "SE"):
                p = bdir / f"img_{iid}_{d}.tif"
                if p.exists():
                    key = "ETD" if d == "SE" else d  # SE is the side detector
                    det[key] = p
            locs.append({"id": iid, "batch": batch, "det": det})
    return locs


def load_gray(path):
    """RGB uint8 SEM image -> float32 grey (median over channels)."""
    im = tifffile.imread(path).astype(np.float32)
    if im.ndim == 3:
        im = np.median(im, axis=2)
    return im


def normalize(img):
    """1st-99.5th percentile normalization to [0,1] (qc_real.py)."""
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ labels
def segment_labels(gray):
    """Faithful copy of qc_real.py:segment() -> labels 0 pore, 1 bulk, 2 bright.

    Pore threshold = lower cut of 3-class multi-Otsu; bright = upper cut,
    opening disk r=2, remove objects < MIN_BRIGHT_PX. No pore cleaning in the
    label itself (same as qc_real.py). Identical settings for every image.
    """
    s = filters.gaussian(normalize(gray), sigma=SIGMA)
    t_dark, t_bright = filters.threshold_multiotsu(s, classes=3)
    bright = s > t_bright
    bright = morphology.binary_opening(bright, morphology.disk(2))
    bright = morphology.remove_small_objects(bright, MIN_BRIGHT_PX)
    labels = np.ones(s.shape, dtype=np.uint8)
    labels[s < t_dark] = 0
    labels[bright] = 2
    return labels


# ------------------------------------------------------------------ features
FEATURE_NAMES = (
    "bse",           # normalized raw grey
    "bse_s1_5",      # gaussian sigma=1.5 (the value the thresholds act on)
    "bse_s4",        # gaussian sigma=4
    "bse_s12",       # gaussian sigma=12 (local background level)
    "bse_rel",       # bse_s1_5 - bse_s12 (brightness relative to surroundings)
    "grad_s1_5",     # gradient magnitude of bse_s1_5 (edges)
    "std_r7",        # local std in radius-7 disk (texture)
    "std_r25",       # local std in radius-25 disk (larger-scale texture)
    "min_r15",       # local min radius-15 (dark-core indicator)
    "max_r15",       # local max radius-15 (bright-core indicator)
    "inl_s1_5",      # Inlens channel, smoothed (topographic contrast)
    "etd_s1_5",      # ETD/SE channel, smoothed (surface contrast)
)


def tile_features(tile_bse, tile_inl, tile_etd):
    """Compute the named feature matrix for one tile -> (H, W, F) float32.

    Same feature set for every tile; feature names are reported next to the
    model's feature importances so the decisions can be questioned.
    """
    from scipy import ndimage as ndi

    bse = normalize(tile_bse)
    s15 = filters.gaussian(bse, SIGMA).astype(np.float32)
    s4 = filters.gaussian(bse, 4.0).astype(np.float32)
    s12 = filters.gaussian(bse, 12.0).astype(np.float32)
    rel = (s15 - s12).astype(np.float32)
    grad = filters.sobel(s15).astype(np.float32)
    mean7 = ndi.uniform_filter(bse, size=15)
    std7 = np.sqrt(np.maximum(ndi.uniform_filter(bse**2, size=15) - mean7**2, 0))
    mean25 = ndi.uniform_filter(bse, size=51)
    std25 = np.sqrt(np.maximum(ndi.uniform_filter(bse**2, size=51) - mean25**2, 0))
    min15 = ndi.grey_erosion(bse, size=(15, 15)).astype(np.float32)
    max15 = ndi.grey_dilation(bse, size=(15, 15)).astype(np.float32)
    inl = filters.gaussian(normalize(tile_inl), SIGMA).astype(np.float32)
    etd = filters.gaussian(normalize(tile_etd), SIGMA).astype(np.float32)
    return np.stack(
        [bse, s15, s4, s12, rel, grad, std7, std25, min15, max15, inl, etd],
        axis=-1,
    ).astype(np.float32)


# ------------------------------------------------------------------ splits
def make_splits(locs):
    """Grouped split: stratified held-out test of whole images + GroupKFold."""
    from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit

    ids = np.array([l["id"] for l in locs])
    batches = np.array([l["batch"] for l in locs])
    # stratified test split via per-batch selection with fixed seed
    rng = np.random.RandomState(SEED)
    test_ids = []
    for b, k in N_TEST_IMAGES_PER_BATCH.items():
        pool = sorted(ids[batches == b])
        test_ids.extend(rng.choice(pool, size=k, replace=False).tolist())
    test_ids = sorted(test_ids)
    train_ids = sorted(i for i in ids if i not in set(test_ids))

    # 5-fold grouped CV over the remaining images, stratified per batch by
    # round-robin assignment after a seeded shuffle.
    folds = [[] for _ in range(N_FOLDS)]
    rng2 = np.random.RandomState(SEED + 1)
    for b in BATCHES:
        pool = [i for i in train_ids if batches[ids.tolist().index(i)] == b]
        pool = sorted(pool)
        rng2.shuffle(pool)
        for j, iid in enumerate(pool):
            folds[j % N_FOLDS].append(iid)
    fold_lists = [sorted(f) for f in folds]
    return {"test": test_ids, "train_val": train_ids, "folds": fold_lists}


def leakage_assert(splits):
    """Sanity check 1: no image ID in more than one split."""
    seen = {}
    ok = True
    for name, ids in [("test", splits["test"])] + [
        (f"fold{k}", f) for k, f in enumerate(splits["folds"])
    ]:
        for iid in ids:
            if iid in seen:
                ok = False
                print(f"LEAK: {iid} in {seen[iid]} and {name}")
            seen[iid] = name
    assert ok, "parent image ID appears in more than one split"
    n = len(seen)
    print(f"leakage check PASS: {n} image IDs, each in exactly one split")
    return ok


# ------------------------------------------------------------------ metrics
def per_class_metrics(y_true, y_pred):
    """Per-class precision/recall/F1/IoU + overall + balanced accuracy."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_IDS))
    rows = {}
    for c, name in zip(CLASS_IDS, CLASS_NAMES):
        tp = cm[c, c]
        pred_c = cm[:, c].sum()
        true_c = cm[c, :].sum()
        union = pred_c + true_c - tp
        rows[name] = {
            "precision": tp / pred_c if pred_c else np.nan,
            "recall": tp / true_c if true_c else np.nan,
            "f1": (2 * tp / (pred_c + true_c)) if (pred_c + true_c) else np.nan,
            "iou": tp / union if union else np.nan,
            "support": int(true_c),
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class": rows,
        "confusion": cm,
    }


def majority_baseline(y_true):
    """Accuracy of predicting the most frequent label class everywhere."""
    vals, counts = np.unique(y_true, return_counts=True)
    return counts.max() / counts.sum()


def get_splits(locs=None):
    """Return the fixed split dict, creating artifacts/splits.json once."""
    ART_DIR.mkdir(parents=True, exist_ok=True)
    path = ART_DIR / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    if locs is None:
        locs = list_locations()
    splits = make_splits(locs)
    splits["seed"] = SEED
    path.write_text(json.dumps(splits, indent=2))
    return splits


def bootstrap_image_ci(per_image_scores, n_boot=2000, seed=SEED):
    """95% CI for the mean of per-image scores, resampling images."""
    rng = np.random.RandomState(seed)
    a = np.asarray(per_image_scores, dtype=float)
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    means = np.array(
        [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n_boot)]
    )
    return float(a.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )
