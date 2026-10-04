"""make_tiles.py - cut every imaged location into a 2x2 grid of tiles.

Each location contributes 4 tiles (<id>_q0.._q3, row-major). BSE, Inlens and
ETD/SE of one location are cut on the SAME grid so the three channels stay
pixel-aligned. The label mask (multi-Otsu segmentation of the full BSE image,
same method as qc_real.py) is generated once per image and cut identically.

Labels come from a threshold pipeline, not a person: model accuracy below
measures agreement with multi-Otsu + morphology, not ground truth.

Outputs (local only, gitignored):
  tiles/<tile_id>.npz   bse/inl/etd/labels uint8 arrays
  tiles/tiles_index.csv tile_id,image_id,batch,quadrant,height,width
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import (
    TILES_DIR,
    list_locations,
    load_gray,
    segment_labels,
)


def run():
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    locs = list_locations()
    print(f"{len(locs)} locations")
    rows = []
    t0 = time.time()
    for li, loc in enumerate(locs):
        imgs = {d: load_gray(loc["det"][d]) for d in ("BSE", "Inlens", "ETD")}
        labels = segment_labels(imgs["BSE"])
        h, w = labels.shape
        h2, w2 = h // 2, w // 2
        quadrants = [
            (slice(0, h2), slice(0, w2)),
            (slice(0, h2), slice(w2, w)),
            (slice(h2, h), slice(0, w2)),
            (slice(h2, h), slice(w2, w)),
        ]
        for q, (rs, cs) in enumerate(quadrants):
            tid = f"{loc['id']}_q{q}"
            np.savez_compressed(
                TILES_DIR / f"{tid}.npz",
                bse=imgs["BSE"][rs, cs].astype(np.uint8),
                inl=imgs["Inlens"][rs, cs].astype(np.uint8),
                etd=imgs["ETD"][rs, cs].astype(np.uint8),
                labels=labels[rs, cs],
            )
            rows.append(
                {
                    "tile_id": tid,
                    "image_id": loc["id"],
                    "batch": loc["batch"],
                    "quadrant": q,
                    "height": rs.stop - rs.start,
                    "width": cs.stop - cs.start,
                }
            )
        print(
            f"[{li + 1}/{len(locs)}] {loc['id']} ({loc['batch']}) "
            f"{h}x{w} -> 4 x {h2}x{w2}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(TILES_DIR / "tiles_index.csv", index=False)
    sz = df[["height", "width"]].describe().loc[["min", "max"]]
    print(f"{len(df)} tiles written to {TILES_DIR}")
    print("tile size px (min/max):", sz.to_dict())
    small = df[(df.height < 200) | (df.width < 200)]
    print(f"tiles under 200px on a side: {len(small)}")
    return df


if __name__ == "__main__":
    run()
