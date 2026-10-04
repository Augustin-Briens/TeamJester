"""Reproduce the accessible-silicon denominator defect (task item A).

Bug: markers.py computes `enclosed` over label(si_part) CONNECTED
COMPONENTS but divides by len(sub) = watershed-separated particles.
Connected components merge touching watershed particles, so the
enclosed share (and si_accessible_frac = 1 - enclosed) is wrong.

Expected (reported): saved 0.523952; watershed-consistent ~0.2455;
area-weighted ~0.3714.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, label

sys.path.insert(0, ".")
from micro2dfn import config, io, objects as obj_, segment


def main(image_id="img_hawkfj64", batch="Batch_3"):
    rec = json.load(open("dfn_output/recipe.json"))
    im = io.load_bse(f"{batch}/{image_id}_BSE.tif", batch=batch)
    seg = segment.segment(im, rec["t_pore"], rec["t_si"])
    si_mask = seg == config.SI
    ob = obj_.extract_objects(im, si_mask)
    ob = obj_.classify_objects(ob, rec["t_core"])
    sub = ob[ob.kind == "si_particle"]
    si_part, _ = obj_.particle_mask(im, si_mask, ob)

    pore = seg == config.PORE
    near_pore = binary_dilation(pore, iterations=config.CONTACT_DILATION_PX)
    si_boundary = si_part & ~binary_erosion(si_part)

    # --- the SHIPPED (buggy) quantity --------------------------------
    lab_cc = label(si_part)
    enclosed_cc = 0
    cov_cc = []
    for r in __import__("skimage.measure", fromlist=["regionprops"]
                        ).regionprops(lab_cc):
        isb = si_boundary[r.coords[:, 0], r.coords[:, 1]]
        nb = int(isb.sum())
        cov = float((isb & near_pore[r.coords[:, 0],
                                   r.coords[:, 1]]).sum() / nb) \
            if nb else 0.0
        cov_cc.append(cov)
        enclosed_cc += cov == 0.0
    n_ws = len(sub)
    saved = 1.0 - enclosed_cc / n_ws
    print(f"watershed particles (sub): {n_ws}")
    print(f"connected components in si_part: {lab_cc.max()}")
    print(f"enclosed components: {enclosed_cc}")
    print(f"SHIPPED  si_accessible_frac = 1 - {enclosed_cc}/{n_ws} "
          f"= {saved:.6f}  (saved CSV: 0.523952)")

    # --- watershed-consistent reconstruction --------------------------
    # rebuild watershed labels restricted to classified particles
    lab_ws = obj_.split_si(si_mask)
    part_labels = set(sub["label"].astype(int))
    enclosed_ws, cov_ws, areas, cov_area = 0, [], [], []
    for lid in part_labels:
        m = lab_ws == lid
        if not m.any():
            continue
        b = (m & ~binary_erosion(m)) & si_boundary
        nb = int(b.sum())
        cov = float((b & near_pore).sum() / nb) if nb else 0.0
        cov_ws.append(cov)
        areas.append(int(m.sum()))
        enclosed_ws += cov == 0.0
    cov_ws = np.asarray(cov_ws)
    areas = np.asarray(areas)
    n_cc = len(cov_ws)
    num_frac = 1.0 - enclosed_ws / max(n_cc, 1)
    aw = float((areas * (cov_ws > 0)).sum() / max(areas.sum(), 1))
    print(f"\nCORRECTED  contacted-particle number fraction = "
          f"1 - {enclosed_ws}/{n_cc} = {num_frac:.6f}  (reported ~0.2455)")
    print(f"CORRECTED  area-weighted contacted share     = {aw:.6f}  "
          f"(reported ~0.3714)")
    print(f"(boundary coverage si_pore_contact = "
          f"{float((si_boundary & near_pore).sum()/si_boundary.sum()):.4f}"
          " — unchanged, different quantity)")


if __name__ == "__main__":
    main()
