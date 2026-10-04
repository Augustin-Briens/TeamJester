"""train.py - dataset assembly + RandomForest training for model_retrain.

Model: sklearn RandomForestClassifier on named per-pixel features
(common.FEATURE_NAMES). Explainable by design: feature importances are
reported; manufacturing users can question each input.

Training data are a RANDOM PIXEL SUBSAMPLE of the training tiles (labels are
the multi-Otsu masks). Geometric augmentation (h/v flips, rot180) is applied
to TRAINING data only, by flipping the already-computed feature maps - for
the symmetric kernels used this equals computing features on the flipped
tile. No augmentation on validation/test. All settings are chosen on the
validation folds only; the test set is evaluated once at the end.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import (
    ART_DIR,
    FEATURE_NAMES,
    RNG_SEED,
    TILES_DIR,
    TRAIN_PX_PER_TILE,
    tile_features,
)

# 60 trees: folds showed the accuracy plateau well below 150, and the
# compressed joblib must stay under GitHub's 100 MB file limit.
RF_KW = dict(
    n_estimators=60,
    max_depth=32,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RNG_SEED,
)

AUGMENT = ("id", "hflip", "vflip", "rot180")


def load_tile(tile_id):
    z = np.load(TILES_DIR / f"{tile_id}.npz")
    return {k: z[k] for k in ("bse", "inl", "etd", "labels")}


def augment_pair(feat, lab, mode):
    if mode == "id":
        return feat, lab
    if mode == "hflip":
        return feat[:, ::-1], lab[:, ::-1]
    if mode == "vflip":
        return feat[::-1], lab[::-1]
    if mode == "rot180":
        return feat[::-1, ::-1], lab[::-1, ::-1]
    raise ValueError(mode)


def tile_index():
    return pd.read_csv(TILES_DIR / "tiles_index.csv")


def tiles_for_images(image_ids):
    idx = tile_index()
    return idx[idx.image_id.isin(set(image_ids))]


def tile_rng(tile_id, seed):
    """Deterministic per-tile RNG so a tile's subsample is reproducible."""
    import zlib

    return np.random.RandomState((zlib.crc32(tile_id.encode()) + seed) % 2**32)


def extract_subsample(tid, px_per_tile, augment=True, seed=RNG_SEED):
    """Compute one tile's feature map, draw a deterministic pixel subsample
    across the augmentation modes, return (X, y) and DROP the feature map
    (full maps are ~170 MB; the subsample is a few MB)."""
    t = load_tile(tid)
    feat = tile_features(t["bse"], t["inl"], t["etd"])
    lab = t["labels"]
    modes = AUGMENT if augment else ("id",)
    rng = tile_rng(tid, seed)
    Xs, ys = [], []
    per_mode = max(1, px_per_tile // len(modes))
    for m in modes:
        f2, l2 = augment_pair(feat, lab, m)
        n = f2.shape[0] * f2.shape[1]
        k = min(per_mode, n)
        sel = rng.choice(n, size=k, replace=False)
        Xs.append(f2.reshape(-1, f2.shape[-1])[sel])
        ys.append(l2.ravel()[sel])
    return np.concatenate(Xs), np.concatenate(ys)


def build_xy(tile_ids, px_per_tile=TRAIN_PX_PER_TILE, augment=True,
             seed=RNG_SEED, cache=None):
    """(X, y) for a list of tiles; cache holds per-tile subsamples so the
    expensive feature computation happens only once per pipeline run."""
    Xs, ys = [], []
    for tid in tile_ids:
        key = (tid, px_per_tile, augment, seed)
        if cache is not None and key in cache:
            X, y = cache[key]
        else:
            X, y = extract_subsample(tid, px_per_tile, augment, seed)
            if cache is not None:
                cache[key] = (X, y)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def train_rf(X, y, seed=RNG_SEED, **overrides):
    kw = dict(RF_KW)
    kw["random_state"] = seed
    kw.update(overrides)
    clf = RandomForestClassifier(**kw)
    clf.fit(X, y)
    return clf


def save_model(clf, path, meta):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "meta": meta}, path, compress=("zlib", 3))


def load_model(path):
    return joblib.load(path)


def predict_labels(clf, feat, chunk_rows=256):
    """Predict a (H,W,F) feature map -> (H,W) uint8 labels, chunked by rows."""
    h, w, f = feat.shape
    out = np.empty((h, w), dtype=np.uint8)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        block = feat[r0:r1].reshape(-1, f)
        out[r0:r1] = clf.predict(block).reshape(r1 - r0, w)
    return out


def predict_proba_labels(clf, feat, chunk_rows=128):
    """Predict labels + max-probability confidence map for a (H,W,F) map."""
    h, w, f = feat.shape
    out = np.empty((h, w), dtype=np.uint8)
    conf = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        block = feat[r0:r1].reshape(-1, f)
        proba = clf.predict_proba(block)
        out[r0:r1] = proba.argmax(1).reshape(r1 - r0, w)
        conf[r0:r1] = proba.max(1).reshape(r1 - r0, w)
    return out, conf
