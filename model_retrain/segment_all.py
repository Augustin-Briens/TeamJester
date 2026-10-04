"""segment_all.py - run the trained RF on every BSE image and save the
per-pixel 3-class segmentation to segmented/.

Outputs per image <id>:
  segmented/<id>_mask.png      3-class colour mask (pore=blue, bulk=grey,
                               bright=orange)
  segmented/<id>_overlay.jpg   mask tinted over the BSE image (half res)
  segmented/<id>_labels.npz    raw uint8 label array (0 pore, 1 bulk, 2 bright)
  segmented/summary.csv        per-image predicted phase fractions + batch
Plus segmented/contact_<batch>.png contact sheets.

Labels are RF predictions reproducing the multi-Otsu pipeline; bright is
ASSUMED silicon-based, grey bulk ASSUMED graphite - appearance only.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import (
    ART_DIR,
    BATCHES,
    OUT_DIR,
    list_locations,
    load_gray,
    normalize,
    tile_features,
)
from train import load_model, predict_labels

SEG_DIR = OUT_DIR / "segmented"
CMAP = ListedColormap(["#35A7FF", "#8C8C8C", "#FF7A00"])
TINT = np.array([[0x35, 0xA7, 0xFF], [0x80, 0x80, 0x80], [0xFF, 0x7A, 0x00]],
                dtype=np.float32) / 255


def run():
    SEG_DIR.mkdir(exist_ok=True)
    clf = load_model(ART_DIR / "rf_model.joblib")["clf"]
    locs = list_locations()
    rows = []
    masks = {}
    t0 = time.time()
    for i, loc in enumerate(locs):
        imgs = {d: load_gray(loc["det"][d]) for d in ("BSE", "Inlens", "ETD")}
        feat = tile_features(imgs["BSE"], imgs["Inlens"], imgs["ETD"])
        pred = predict_labels(clf, feat)
        iid = loc["id"]
        np.savez_compressed(SEG_DIR / f"{iid}_labels.npz", labels=pred)
        plt.imsave(SEG_DIR / f"{iid}_mask.png", pred, cmap=CMAP, vmin=0,
                   vmax=2)
        bse = normalize(imgs["BSE"])
        rgb = np.stack([bse] * 3, axis=-1)
        tint = TINT[pred]
        over = np.clip(0.6 * rgb + 0.4 * tint, 0, 1)[::2, ::2]
        plt.imsave(SEG_DIR / f"{iid}_overlay.jpg", over,
                   pil_kwargs={"quality": 85})
        fr = {n: float((pred == c).mean()) for c, n in
              ((0, "pore"), (1, "bulk"), (2, "bright"))}
        rows.append({"image_id": iid, "batch": loc["batch"], **fr})
        masks[iid] = pred
        print(f"[{i + 1}/{len(locs)}] {iid} {fr} ({time.time() - t0:.0f}s)",
              flush=True)
    pd.DataFrame(rows).to_csv(SEG_DIR / "summary.csv", index=False)
    for b in BATCHES:
        ids = [r["image_id"] for r in rows if r["batch"] == b]
        n = len(ids)
        cols = 4
        rr = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rr, cols, figsize=(16, 3.2 * rr))
        axes = np.atleast_2d(axes)
        for j in range(rr * cols):
            ax = axes[j // cols, j % cols]
            ax.axis("off")
            if j < n:
                ax.imshow(masks[ids[j]][::4, ::4], cmap=CMAP, vmin=0, vmax=2)
                ax.set_title(ids[j], fontsize=7)
        fig.suptitle(f"{b} - RF predicted phase masks "
                     "(pore=blue bulk=grey bright=orange)")
        fig.tight_layout()
        fig.savefig(SEG_DIR / f"contact_{b}.png", dpi=150)
        plt.close(fig)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
