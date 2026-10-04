#!/usr/bin/env python3
"""Build dataset_manifest.csv — one row per BSE field of view.

Every specimen/image/channel in the dataset is inventoried.  Detector
channels are recorded as columns, NOT counted as separate specimens:
the three channels are registered views of the same field of view.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "teaching_micro2dfn", "dataset_manifest.csv")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_size_um(path: str) -> float:
    try:
        with tifffile.TiffFile(path) as t:
            tags = t.pages[0].tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                unit = tags["ResolutionUnit"].value
                if unit == 2:
                    return 25_400.0 / (num / den)
                if unit == 3:
                    return 10_000.0 / (num / den)
    except Exception:
        pass
    return 0.025


def main() -> None:
    rows = []
    recipe_md5 = {}
    rpath = os.path.join(ROOT, "validated_comparison", "recipe.json")
    if os.path.exists(rpath):
        recipe_md5 = json.load(open(rpath)).get("file_md5", {})

    batches = ["Batch_1", "Batch_2", "Batch_3", "New_Images_Batch"]
    for batch in batches:
        bdir = os.path.join(ROOT, batch)
        if not os.path.isdir(bdir):
            continue
        bses = sorted(f for f in os.listdir(bdir) if f.endswith("_BSE.tif"))
        for fn in bses:
            iid = fn.replace("_BSE.tif", "")
            p = os.path.join(bdir, fn)
            with tifffile.TiffFile(p) as t:
                h, w = t.pages[0].shape[:2]
            chans = {}
            for det in ("Inlens", "ETD", "SE"):
                cp = os.path.join(bdir, f"{iid}_{det}.tif")
                chans[det] = os.path.exists(cp)
            rows.append(dict(
                batch=batch, image_id=iid,
                file=os.path.relpath(p, ROOT),
                width_px=w, height_px=h,
                pixel_um=round(pixel_size_um(p), 7),
                width_um=round(w * pixel_size_um(p), 1),
                height_um=round(h * pixel_size_um(p), 1),
                has_inlens=int(chans["Inlens"]),
                has_etd=int(chans["ETD"]),
                has_se=int(chans["SE"]),
                md5_bse=md5(p),
                recipe_md5_match=str(
                    recipe_md5.get(iid, "") == md5(p)
                    if iid in recipe_md5 else "n/a"),
            ))
    df = pd.DataFrame(rows)
    df["independent_unit"] = "field_of_view"
    df["note"] = np.where(
        df.batch.eq("New_Images_Batch"),
        "unlabelled new images — batch unknown by design",
        "")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} fields of view, "
          f"{df[['has_inlens','has_etd','has_se']].sum().to_dict()} "
          f"second channels")


if __name__ == "__main__":
    sys.exit(main())
