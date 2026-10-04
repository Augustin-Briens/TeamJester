#!/usr/bin/env python3
"""recon3d — optional 2-D->3-D reconstruction & transport branch.

    .venv/bin/python run_recon3d.py --fast           # CPU smoke: V=64,
                                                     300 iters — sanity
                                                     only, NOT trained
    .venv/bin/python run_recon3d.py                  # local full run
    .venv/bin/python run_recon3d.py --analyze-only   # redo transport +
                                                     report on cached
                                                     volumes
    .venv/bin/python run_recon3d.py --masks-only     # build & cache the
                                                     frozen training set

GPU training is meant to run on Modal:

    .venv/bin/python -m modal run modal_recon3d.py               # full
    .venv/bin/python -m modal run modal_recon3d.py --fast        # dev
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recon3d import config, data, analysis


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.dirname(
        os.path.abspath(__file__)))
    ap.add_argument("--out", default="recon3d_output")
    ap.add_argument("--batches", default=None,
                    help="comma list; default: all Batch_* dirs")
    ap.add_argument("--fast", action="store_true",
                    help="V=64, 300 iters, 1 seed x 2 vols — dev only")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--volume", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="e.g. 0,1,2")
    ap.add_argument("--vols", type=int, default=None,
                    help="volumes per seed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--downsample", type=int, default=None,
                    help="mask downsample factor; default config (4 = "
                         "100 nm/voxel CPU build, use 2 = 50 nm for "
                         "V=128 GPU runs)")
    ap.add_argument("--baseline", default="Batch_3",
                    help="reference batch the measuring ruler is "
                         "fitted on (reference-only calibration)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--masks-only", action="store_true")
    return ap.parse_args()


def get_batches(root: str, arg: str | None) -> list[str]:
    if arg:
        return arg.split(",")
    from micro2dfn import io
    return sorted(io.discover_batches(root))


def build_or_load_masks(root, out, batches, downsample,
                        baseline="Batch_3"):
    """Masks + ruler provenance.  Cache is keyed on the ruler hash so
    masks built under a different ruler are never silently reused."""
    t_pore, t_si, prov = data.load_or_fit_recipe(root, out, baseline)
    tag = data.ruler_tag(prov)
    prov["tag"] = tag
    print(f"ruler {tag}: t_pore={t_pore:.4f} t_si={t_si:.4f} "
          f"(fitted on {prov['fitted_on']}, source {prov['source']})")
    cache = os.path.join(out, f"masks_{tag}_ds{downsample}.npz")
    if os.path.exists(cache):
        masks = data.unpack_masks(open(cache, "rb").read())
        missing = [b for b in batches if b not in masks]
        if not missing:
            print(f"masks: reusing {cache}")
            return masks, prov
        print(f"masks: cache missing {missing} — rebuilding")
    else:
        masks = {}
    print(f"masks: segmenting {batches} at 25 nm -> {downsample}x "
          "majority vote...")
    t0 = time.time()
    masks.update(data.batch_masks(root, batches, (t_pore, t_si),
                                  downsample))
    blob = data.pack_masks(masks)
    with open(cache, "wb") as fh:
        fh.write(blob)
    data.save_manifest(masks, os.path.join(out, "masks_manifest.json"),
                       prov)
    n = {b: len(v) for b, v in masks.items()}
    print(f"  {n} images in {time.time() - t0:.0f}s -> {cache}")
    return masks, prov


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    batches = get_batches(a.data, a.batches)
    fast = a.fast
    V = a.volume or (64 if fast else config.VOLUME)
    iters = a.iters or (config.ITERS_FAST if fast else config.ITERS)
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds else
             list(config.SEEDS_FAST if fast else config.SEEDS))
    vols = a.vols or (config.VOLUMES_FAST if fast
                      else config.VOLUMES_PER_SEED)
    ds = a.downsample or config.DOWNSAMPLE
    masks, prov = build_or_load_masks(a.data, a.out, batches, ds,
                                      a.baseline)
    tag = prov["tag"]
    settings = dict(V=V, voxel_nm=25.0 * ds,
                    iters=iters, seeds=seeds, vols_per_seed=vols,
                    n_seeds=len(seeds),
                    n_volumes=len(seeds) * vols, mode="local",
                    fast=bool(fast), downsample=ds,
                    baseline=a.baseline, ruler=tag,
                    ruler_info={k: v for k, v in prov.items()
                                if k != "tag"})
    with open(os.path.join(a.out, "run_settings.json"), "w") as fh:
        json.dump(settings, fh, indent=1)
    if a.masks_only:
        return

    if not a.analyze_only:
        from recon3d import slicegan
        import torch
        device = a.device or ("cuda" if torch.cuda.is_available() else
                              "mps" if getattr(torch.backends, "mps",
                                               None)
                              and torch.backends.mps.is_available()
                              else "cpu")
        settings["device"] = device
        with open(os.path.join(a.out, "run_settings.json"),
                  "w") as fh:
            json.dump(settings, fh, indent=1)
        print(f"training locally on {device}: V={V}, {iters} iters, "
              f"seeds={seeds}, vols/seed={vols}")
        if device == "cpu" and not fast:
            print("  WARNING: CPU training of the full config is very "
                  "slow — use --fast or Modal.")
        for b in batches:
            for s in seeds:
                mtag = f"{b}__s{s}__V{V}i{iters}__r{tag}"
                npz = os.path.join(a.out, "volumes", f"{mtag}.npz")
                if os.path.exists(npz):
                    print(f"  {mtag}: cached, skipping")
                    continue
                print(f"  {mtag}: training ({iters} iters)...",
                      flush=True)
                state = slicegan.train(masks[b], seed=s, iters=iters,
                                       V=V, device=device)
                v = slicegan.generate(state, vols, seed=1000 + s,
                                      device=device)
                slicegan.save_volumes(
                    v, npz, dict(batch=b, seed=s, V=V, iters=iters,
                                 ruler=tag))
                pd.DataFrame(state["hist"]).assign(
                    seconds=state["hist"]["train_seconds"]).to_csv(
                    os.path.join(a.out, "models", f"{mtag}.csv"),
                    index=False)
    else:
        print("analyze-only: reusing volumes")

    analysis.run_analysis(a.out, batches, masks, settings)


if __name__ == "__main__":
    main()
