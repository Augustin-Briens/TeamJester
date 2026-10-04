"""Assign New_Images_Batch photos to existing batches.

Method (interpretable, declared up front — v2 after envelope-coverage
fix):
  1. Extract the same 9 primary features + diagnostics as the frozen
     vcompare recipe (thresholds/classifier fitted on Batch_3 only).
  2. PRIMARY RULE — envelope coverage: for each batch, count how many
     of the 9 features fall inside that batch's observed min-max range.
     - exactly one batch contains the image fully (9/9) -> that batch
     - more than one batch contains it fully -> inconclusive_between
     - no batch contains it fully -> the batch with most features
       inside wins, provided its median robust-z stays below the OOD
       threshold; otherwise out_of_distribution.
  3. Secondary scores kept for transparency: median robust z
     |x - median| / (1.4826·MAD) per batch (biased toward wide batches —
     coverage is the fairer primary rule), and per-feature z table.
  4. No image is forced into a batch on weak evidence.

Outputs: new_image_assignment/{assignments.csv, feature_table.csv,
assignment_report.md, fig_assignment.png}
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from vcompare import assemble, config, extract, recipe as recipe_mod

RECIPE = "validated_comparison/recipe.json"
FEATURES = list(config.PRIMARY_FEATURES)
# feature direction note for the report
Z_THRESH = 2.5          # median-z above this => no batch contains the point
EPS = 1e-9


def load_existing_features() -> pd.DataFrame:
    df = pd.read_csv("validated_comparison/per_image_features.csv")
    return df[df["batch"].isin(("Batch_1", "Batch_2", "Batch_3"))]


def extract_new(folder: str, recipe: dict, cache: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, "*_BSE.tif")))
    assert paths, f"no *_BSE.tif under {folder}"
    extract.set_thresholds(recipe["t_pore"], recipe["t_si"])
    extractions, objects = {}, []
    for p in paths:
        print(f"  extracting {os.path.basename(p)}", flush=True)
        ext = extract.extract_cached(p, "New_Images_Batch", cache)
        extractions[ext["meta"]["image_id"]] = ext
        objects.append(ext["objects"])
    ob = pd.concat(objects, ignore_index=True)
    ob = assemble.classify_objects(ob, recipe["t_core"])
    feats = assemble.build_features(extractions, ob, recipe)
    return feats


def batch_stats(existing: pd.DataFrame) -> dict:
    stats = {}
    for b, sub in existing.groupby("batch"):
        stats[b] = {
            f: {"med": float(sub[f].median()),
                "mad": float(1.4826 * np.median(
                    np.abs(sub[f] - sub[f].median()))),
                "lo": float(sub[f].min()),
                "hi": float(sub[f].max())}
            for f in FEATURES
        }
    return stats


def assign(row: pd.Series, stats: dict) -> dict:
    out = {"image_id": row["image_id"]}
    z_per_batch, inside = {}, {}
    for b, bs in stats.items():
        zs = []
        for f in FEATURES:
            mad = max(bs[f]["mad"], EPS)
            zs.append(abs(row[f] - bs[f]["med"]) / mad)
            out[f"z_{b}_{f}"] = zs[-1]
        z_per_batch[b] = float(np.median(zs))
        out[f"score_{b}"] = z_per_batch[b]
        inside[b] = int(sum(bs[f]["lo"] <= row[f] <= bs[f]["hi"]
                            for f in FEATURES))
        out[f"inside_env_{b}"] = inside[b]
        out[f"outside_env_{b}"] = len(FEATURES) - inside[b]

    # primary rule: envelope coverage (unbiased by batch spread)
    full = [b for b, n in inside.items() if n == len(FEATURES)]
    if len(full) == 1:
        out["assigned_batch"] = full[0]
        out["confidence"] = ("envelope-consistent "
                             f"(9/9 features inside {full[0]}; "
                             + "; ".join(f"{b} {n}/9" for b, n
                                         in inside.items()
                                         if b != full[0]) + ")")
    elif len(full) > 1:
        out["assigned_batch"] = "inconclusive_between_" + \
            "_".join(b[-1] for b in full)
        out["confidence"] = "inside multiple batch envelopes — " \
            "the batches overlap and this image cannot be separated"
    else:
        best = max(inside, key=inside.get)
        tied = [b for b, n in inside.items() if n == inside[best]]
        if len(tied) > 1:
            order = sorted(z_per_batch, key=z_per_batch.get)
            best = order[0]
            tied_note = f"coverage tie {tied} -> lowest median-z"
        else:
            tied_note = ""
        if z_per_batch[best] > Z_THRESH:
            out["assigned_batch"] = "out_of_distribution"
            out["confidence"] = "none — outside every batch envelope"
        else:
            out["assigned_batch"] = best
            out["confidence"] = (
                f"best envelope coverage {inside[best]}/9 "
                f"(median-z {z_per_batch[best]:.2f}) {tied_note}").strip()
    order = sorted(z_per_batch, key=z_per_batch.get)
    out["best_score"] = z_per_batch[order[0]]
    out["second_best"] = order[1]
    out["score_margin"] = (z_per_batch[order[1]]
                           - z_per_batch[order[0]]) / max(
                               z_per_batch[order[0]], EPS)
    out["medianz_winner"] = order[0]   # kept for bias transparency
    return out


def main(folder: str = "New_Images_Batch",
         recipe_path: str = RECIPE,
         out_dir: str = "new_image_assignment",
         cache: str = "new_image_assignment/cache"):
    os.makedirs(out_dir, exist_ok=True)
    recipe = recipe_mod.load_recipe(recipe_path)
    print(f"recipe: fitted on {recipe['fitted_on']} "
          f"(t_pore={recipe['t_pore']:.4f}, t_si={recipe['t_si']:.4f}, "
          f"t_core={recipe['t_core']:.4f})")

    existing = load_existing_features()
    new = extract_new(folder, recipe, cache)
    new.to_csv(os.path.join(out_dir, "feature_table.csv"), index=False)

    stats = batch_stats(existing)
    rows = [assign(r, stats) for _, r in new.iterrows()]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "assignments.csv"), index=False)

    # ---- report ------------------------------------------------------
    L = ["# New-image batch assignment — ALL CALLS: \"CAN'T TELL\"\n",
         "**Status (acquisition analysis):** each new photo's frame "
         "height occurs in only one batch's acquisition group, and "
         "each call below followed that session — the assignments "
         "cannot separate batch-match from session-match. The "
         "protocol (predictions saved blind before any ground truth) "
         "is kept; the verdicts are relabelled \"can't tell\" until "
         "session-controlled data exists. Do not quote the assigned "
         "column as material evidence.\n",
         f"Method (v2): primary rule = envelope coverage — count of the "
         f"{len(FEATURES)} declared features inside each batch's "
         "observed min-max range (unbiased by batch spread). Exactly "
         "one batch containing 9/9 => that batch; several => "
         "inconclusive; none => best coverage wins unless median "
         f"robust-z >{Z_THRESH} => out-of-distribution. Median-z kept "
         "as a secondary score (it favours wide batches — see "
         "`medianz_winner` in assignments.csv). "
         f"Recipe frozen on Batch_3 (`{recipe_path}`).\n",
         "| image | assigned | inside env. (B1/B2/B3) | median-z best "
         "| confidence |",
         "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['image_id']} | {r['assigned_batch']} | "
                 f"{r['inside_env_Batch_1']}/{r['inside_env_Batch_2']}"
                 f"/{r['inside_env_Batch_3']} | {r['medianz_winner']} "
                 f"{r['best_score']:.2f} | {r['confidence'][:60]} |")
    L.append("\n## Per-image notes\n")
    for _, r in res.iterrows():
        nr = new[new.image_id == r["image_id"]].iloc[0]
        L.append(f"### {r['image_id']} → {r['assigned_batch']} "
                 f"({r['confidence']})")
        for b in ("Batch_1", "Batch_2", "Batch_3"):
            L.append(f"- vs {b}: {r[f'inside_env_{b}']}/9 features "
                     f"inside envelope; median-z "
                     f"{r[f'score_{b}']:.2f}")
        ref_batch = r["assigned_batch"] if r["assigned_batch"] in stats \
            else r["medianz_winner"]
        drv = sorted(((f, r[f"z_{ref_batch}_{f}"]) for f in FEATURES),
                     key=lambda t: -t[1])[:3]
        L.append("- largest deviations vs " + ref_batch + ": " +
                 "; ".join(f"{f} z={z:.1f}" for f, z in drv))
        L.append(f"- key features: pore_frac={nr['pore_frac']:.3f}, "
                 f"pores/mpx={nr['pores_per_mpx']:.0f} "
                 f"(noise-adj {nr['pores_per_mpx_noise_adj']:.0f}), "
                 f"Si cand={nr['si_candidate_frac']:.3f}, "
                 f"uncertain={nr['uncertain_bright_frac']:.3f}, "
                 f"Si d50={nr['si_d50_um']:.2f} um, "
                 f"R={nr['si_clustering_R']:.2f}\n")
    L.append("## Caveats\n")
    L.append("- Assignment is similarity, not provenance: batches "
             "overlap on every feature, so weak-confidence calls are "
             "expected for images near the overlap region.")
    L.append("- With 7 images in Batch_1/2 and 17 in Batch_3, envelope "
             "estimates are noisy; scores carry that uncertainty.")
    L.append("- The frozen recipe segments BSE only; InLens/ETD "
             "channels are recorded but not used (channel comparability "
             "not yet established).")
    L.append("- Session confound: the calls follow each photo's "
             "acquisition group; they measure similarity to that "
             "session's batch, not provenance. Verdicts are \"can't "
             "tell\" until controlled re-imaging exists.")
    # ---- figure: z-heatmap of new images vs batch envelopes ---------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 2.4))
        mat = np.array([[r[f"z_{b}_{f}"] for f in FEATURES]
                        for _, r in res.iterrows()
                        for b in ("Batch_1", "Batch_2", "Batch_3")])
        ylab = [f"{r['image_id']} vs {b}"
                for _, r in res.iterrows()
                for b in ("Batch_1", "Batch_2", "Batch_3")]
        im_ = ax.imshow(np.clip(mat, 0, 5), aspect="auto",
                        cmap="magma_r", vmin=0, vmax=5)
        ax.set_yticks(range(len(ylab)), ylab, fontsize=7)
        ax.set_xticks(range(len(FEATURES)),
                      [f.replace("pores_per_mpx_noise_adj", "pore/mpx·nadj")
                       for f in FEATURES], rotation=45, ha="right",
                      fontsize=7)
        ax.set_title("|z| vs each batch envelope (clipped at 5)")
        fig.colorbar(im_, ax=ax, label="robust |z|")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_assignment.png"),
                    dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"  (figure skipped: {e})")

    path = os.path.join(out_dir, "assignment_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:20]))
    print(f"\n-> {path}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
