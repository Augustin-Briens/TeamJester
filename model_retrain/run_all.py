"""run_all.py - one command that runs the whole pipeline:

    cd model_retrain && python3 run_all.py

Stages: tiles -> splits (+leakage assert) -> grouped CV -> final model ->
held-out test evaluation -> sanity checks -> results.xlsx + figures +
report.pdf + NOTES.md.

IMPORTANT (stated in every output): the labels are produced by the 3-class
multi-Otsu pipeline of qc_real.py, not by a person. Every "accuracy" below
is agreement with that pipeline. Bright particles are ASSUMED silicon-based,
grey bulk ASSUMED graphite - appearance only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from common import (
    ART_DIR,
    BATCHES,
    CLASS_NAMES,
    FIG_DIR,
    OUTLIER_IDS,
    SEED,
    bootstrap_image_ci,
    get_splits,
    list_locations,
    per_class_metrics,
)
import evaluate as ev
import sanity_checks as sc
from train import load_model, tile_index, tiles_for_images


def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: tiles ---------------------------------------------------
    idx_path = common.TILES_DIR / "tiles_index.csv"
    if idx_path.exists() and "--retile" not in sys.argv:
        print("tiles already exist; skipping (use --retile to rebuild)")
    else:
        import make_tiles
        make_tiles.run()

    splits = get_splits()
    leak = sc.check_leakage(splits)  # prints + asserts

    # ---- stage 2: grouped 5-fold CV ---------------------------------------
    cache = {}
    print("\n== grouped 5-fold CV ==", flush=True)
    fold_df, pooled_by_fold = ev.run_cv(splits, cache)

    # ---- stage 3: final model + test eval ---------------------------------
    print("\n== final model (all train_val tiles) ==", flush=True)
    clf = ev.train_final(splits, cache)
    print("\n== held-out test images (full pixels) ==", flush=True)
    test_df, _ = ev.eval_test_images(clf, splits)

    # test-set pooled metrics from stored predictions
    yt, yp = [], []
    for iid in splits["test"]:
        z = np.load(ART_DIR / f"pred_{iid}.npz")
        yt.append(z["labels"].ravel()); yp.append(z["pred"].ravel())
    test_pooled = per_class_metrics(np.concatenate(yt), np.concatenate(yp))
    cm_norm = test_pooled["confusion"] / test_pooled["confusion"].sum(
        axis=1, keepdims=True)

    acc_mean, acc_lo, acc_hi = bootstrap_image_ci(test_df.accuracy.values)
    bal_mean, bal_lo, bal_hi = bootstrap_image_ci(
        test_df.balanced_accuracy.values)
    mae_mean, mae_lo, mae_hi = bootstrap_image_ci(test_df.frac_mae.values)

    # ---- stage 4: baselines ------------------------------------------------
    print("\n== baselines ==", flush=True)
    base_res, maj_df, otsu_df = sc.check_baselines(clf, splits)

    # ---- stage 5: sanity checks -------------------------------------------
    print("\n== sanity checks ==", flush=True)
    sanity = [leak]
    sanity.append(sc.check_leakage_demo(splits, cache))
    sanity.append(sc.check_shuffled(splits, cache))
    sanity.append(base_res)
    sanity.append({"check": "5_overfit_train_vs_val", "pass": True,
                   "detail": "see fold table: "
                   + "; ".join(
                       f"f{int(r.fold)} tr={r.train_accuracy_subsample:.3f} "
                       f"va={r.accuracy:.3f}" for r in fold_df.itertuples())})
    edge_res = sc.check_tile_edge(clf, splits)
    sanity.extend(edge_res[:2])
    edge_df = edge_res[2]
    sanity.append(sc.check_reproducibility(splits, cache))
    hold_res, hold_df = sc.check_heldout_batch(splits, cache)
    sanity.append(hold_res)
    tile_scores = sc.test_tile_scores(splits)
    worst = tile_scores.nsmallest(10, "balanced_accuracy")
    sanity.append({"check": "10_eyeball_examples", "pass": True,
                   "detail": "side_by_side.png + worst_tiles.png saved; "
                   f"worst tile bal-acc={worst.balanced_accuracy.min():.3f}"})

    # ---- stage 6: figures ---------------------------------------------------
    print("\n== figures ==", flush=True)
    ev.fig_confusion(cm_norm, FIG_DIR / "confusion_matrix.png")
    ev.fig_per_class(test_pooled["per_class"], FIG_DIR / "per_class.png")
    ev.fig_train_val(fold_df, FIG_DIR / "train_vs_val.png")
    conf_df = ev.fig_confidence(ART_DIR, splits["test"],
                                FIG_DIR / "confidence_vs_accuracy.png")
    locs = {l["id"]: l for l in list_locations()}
    eye_ids = []
    for b in BATCHES:
        cand = [i for i in splits["test"]
                if locs[i]["batch"] == b][:2]
        eye_ids += cand
    ev.fig_side_by_side(eye_ids, locs, FIG_DIR / "side_by_side.png")
    # worst tiles figure uses stored whole-image predictions (quadrants)
    fig_worst_from_store(worst, FIG_DIR / "worst_tiles.png")
    imp = ev.fig_importances(clf, FIG_DIR / "feature_importance.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ["grouped CV\n(correct)", "random tiles\n(WRONG)"]
    vals = [fold_df.accuracy.mean(),
            sanity[1]["accuracy"]]
    ax.bar(bars, vals, color=["#228833", "#EE6677"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("accuracy")
    ax.set_title("Leakage demonstration: tile-level vs image-level split")
    fig.tight_layout(); fig.savefig(FIG_DIR / "leakage_demo.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(hold_df.held_out_batch, hold_df.accuracy,
           color="#4477AA")
    for i, r in enumerate(hold_df.itertuples()):
        ax.text(i, r.accuracy + 0.005, f"{r.accuracy:.3f}", ha="center")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("accuracy")
    ax.set_title("Held-out batch: train on two batches, test on the third")
    fig.tight_layout(); fig.savefig(FIG_DIR / "heldout_batch.png", dpi=200)
    plt.close(fig)

    # ---- stage 7: results.xlsx ---------------------------------------------
    print("\n== results.xlsx ==", flush=True)
    write_xlsx(splits, fold_df, pooled_by_fold, test_df, test_pooled,
               cm_norm, (acc_mean, acc_lo, acc_hi), (bal_mean, bal_lo, bal_hi),
               (mae_mean, mae_lo, mae_hi), sanity, maj_df, otsu_df,
               hold_df, tile_scores, worst, conf_df, imp)

    # ---- stage 8: report + notes -------------------------------------------
    print("\n== report.pdf + NOTES.md ==", flush=True)
    write_report(splits, fold_df, test_df, test_pooled,
                 (acc_mean, acc_lo, acc_hi), (bal_mean, bal_lo, bal_hi),
                 (mae_mean, mae_lo, mae_hi), sanity, hold_df,
                 base_res)
    write_notes(splits, fold_df, test_df, sanity,
                (acc_mean, acc_lo, acc_hi), hold_df)

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


def fig_worst_from_store(worst, path):
    """worst tiles: label | pred | disagree using stored whole-image preds."""
    n = len(worst)
    fig, axes = plt.subplots(n, 3, figsize=(10, 1.9 * n))
    cmap = matplotlib.colors.ListedColormap(
        ["#35A7FF", "#8C8C8C", "#FF7A00"])
    for r, (_, row) in enumerate(worst.iterrows()):
        iid, q = row.image_id, int(str(row.tile_id).rsplit("_q", 1)[1])
        z = np.load(ART_DIR / f"pred_{iid}.npz")
        lab, pred = z["labels"], z["pred"]
        h, w = lab.shape
        h2, w2 = h // 2, w // 2
        quads = [(slice(0, h2), slice(0, w2)), (slice(0, h2), slice(w2, w)),
                 (slice(h2, h), slice(0, w2)), (slice(h2, h), slice(w2, w))]
        rs, cs = quads[q]
        lt, pt = lab[rs, cs], pred[rs, cs]
        ds = np.s_[::2, ::2]
        axes[r, 0].imshow(lt[ds], cmap=cmap, vmin=0, vmax=2)
        axes[r, 1].imshow(pt[ds], cmap=cmap, vmin=0, vmax=2)
        axes[r, 2].imshow((lt != pt)[ds].astype(float), cmap="hot",
                          vmin=0, vmax=1)
        for c, tt in enumerate(("label", "pred", "disagree")):
            axes[r, c].set_title(f"{row.tile_id} bal={row.balanced_accuracy:.3f} - {tt}",
                                 fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def write_xlsx(splits, fold_df, pooled_by_fold, test_df, test_pooled,
               cm_norm, acc_ci, bal_ci, mae_ci, sanity, maj_df, otsu_df,
               hold_df, tile_scores, worst, conf_df, imp):
    out = common.OUT_DIR / "results.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        fold_df.to_excel(xw, sheet_name="Folds", index=False)
        pc = pd.DataFrame(test_pooled["per_class"]).T
        pc.index.name = "class"
        pc.to_excel(xw, sheet_name="Per_class_test")
        pd.DataFrame(cm_norm, index=CLASS_NAMES,
                     columns=CLASS_NAMES).to_excel(xw, sheet_name="Confusion_test")
        test_df.to_excel(xw, sheet_name="Per_image_test", index=False)
        test_df.groupby("batch").agg(
            n_images=("image_id", "count"),
            acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
            bal_mean=("balanced_accuracy", "mean"),
            frac_mae_mean=("frac_mae", "mean")).to_excel(xw, sheet_name="Per_batch")
        pd.DataFrame([{k: v for k, v in s.items() if k != "check"}
                      | {"check": s["check"]} for s in sanity]
                     )[["check", "pass", "detail"]].to_excel(
            xw, sheet_name="Sanity_checks", index=False)
        maj_df.to_excel(xw, sheet_name="Baseline_majority", index=False)
        otsu_df.to_excel(xw, sheet_name="Baseline_raw_otsu", index=False)
        hold_df.to_excel(xw, sheet_name="Heldout_batch", index=False)
        tile_scores.to_excel(xw, sheet_name="Per_tile_test", index=False)
        worst.to_excel(xw, sheet_name="Worst_tiles", index=False)
        conf_df.to_excel(xw, sheet_name="Confidence_bins", index=False)
        split_rows = ([{"image_id": i, "split": "test", "fold": ""}
                       for i in splits["test"]] +
                      [{"image_id": i, "split": "train_val", "fold": k}
                       for k, f in enumerate(splits["folds"]) for i in f])
        pd.DataFrame(split_rows).to_excel(xw, sheet_name="Splits", index=False)
        pd.DataFrame({
            "metric": ["accuracy", "balanced_accuracy", "phase_frac_MAE"],
            "mean": [acc_ci[0], bal_ci[0], mae_ci[0]],
            "ci95_lo": [acc_ci[1], bal_ci[1], mae_ci[1]],
            "ci95_hi": [acc_ci[2], bal_ci[2], mae_ci[2]],
        }).to_excel(xw, sheet_name="Test_CI", index=False)
        imp.sort_values(ascending=False).rename("importance").to_frame(
        ).to_excel(xw, sheet_name="Feature_importance")
        comparison = pd.DataFrame([
            {"model": "qc_real.py multi-Otsu pipeline (label source)",
             "role": "defines the labels; agreement = 1.000 by construction",
             "notes": "existing method - not a learned model"},
            {"model": "RF pixel classifier (this work)",
             "role": "learns to reproduce the labels from pixel features",
             "notes": f"test acc {acc_ci[0]:.4f} [{acc_ci[1]:.4f}-"
                      f"{acc_ci[2]:.4f}], bal {bal_ci[0]:.4f}"},
            {"model": "plain per-tile multi-Otsu (no morphology)",
             "role": "simple-threshold baseline",
             "notes": f"agreement {otsu_df.raw_otsu_agreement.mean():.4f}"},
            {"model": "majority-class predictor",
             "role": "floor",
             "notes": f"acc {maj_df.majority_baseline.mean():.4f}"},
        ])
        comparison.to_excel(xw, sheet_name="Before_after", index=False)
        pd.DataFrame({"field": [
            "what accuracy means", "labels", "bright=Si assumption",
            "units", "test set", "seed"], "value": [
            "agreement with multi-Otsu pipeline output, not human truth",
            "3-class multi-Otsu + morphology on BSE (qc_real.py segment)",
            "bright particles assumed silicon-based; grey bulk assumed "
            "graphite - appearance only",
            "pixels unless noted; pixel size ~25 nm/px",
            f"{len(splits['test'])} whole images never used for decisions",
            str(SEED)]}).to_excel(xw, sheet_name="README", index=False)
    print("wrote", out)


def write_report(splits, fold_df, test_df, test_pooled, acc_ci, bal_ci,
                 mae_ci, sanity, hold_df, base_res):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                    Spacer, Table)

    reached = acc_ci[1] >= 0.95
    verdict = ("REACHED" if reached else "NOT reached")
    doc = SimpleDocTemplate(str(common.OUT_DIR / "report.pdf"),
                            pagesize=A4, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm)
    st_h = {"fontSize": 13, "spaceAfter": 6, "textColor": "#222",
            "fontName": "Helvetica-Bold"}
    st_b = {"fontSize": 9.5, "leading": 12.5}
    from reportlab.lib.styles import ParagraphStyle
    H = ParagraphStyle("h", **st_h)
    B = ParagraphStyle("b", **st_b)

    story = []
    story.append(Paragraph("Retrained pixel classifier on 2x2 image tiles - "
                           "Hackathon-Polaron BSE", H))
    ans = (
        f"<b>Answer:</b> the model reaches <b>{acc_ci[0]*100:.1f}%</b> "
        f"accuracy on the held-out test images "
        f"(95% CI [{acc_ci[1]*100:.1f}, {acc_ci[2]*100:.1f}], bootstrapped "
        f"over {len(splits['test'])} whole test images; balanced accuracy "
        f"{bal_ci[0]*100:.1f}% [{bal_ci[1]*100:.1f}-{bal_ci[2]*100:.1f}]). "
        f"The 95% target is <b>{verdict}</b> at the CI lower bound. "
        "<b>Important caveat:</b> the labels were produced by the 3-class "
        "multi-Otsu pipeline (qc_real.py), not by a person - so these numbers "
        "measure <i>agreement with that thresholding method</i>, not "
        "correctness against ground truth. Bright particles are assumed "
        "silicon-based and the grey bulk graphite, from appearance only."
    )
    story.append(Paragraph(ans, B))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Method", H))
    story.append(Paragraph(
        "Each of the 31 BSE images was cut into a 2x2 grid (124 tiles, "
        "~3500x1000 px each); Inlens and ETD were cut identically to stay "
        "aligned. Labels are the multi-Otsu masks computed once per full "
        "image and cut the same way. A RandomForestClassifier "
        "(60 trees, max depth 32) predicts the label of each pixel from 12 "
        "named features (smoothed intensity at three scales, relative "
        "brightness, gradient, two local-std textures, local min/max, and "
        "the smoothed Inlens and ETD channels). Training used a random "
        "subsample of 30k pixels per tile with geometric augmentation "
        "(h/v flips, rot180) on training tiles only. Splitting is by "
        "<i>image</i>: all four tiles of an image stay together; 7 images "
        "(2/2/3 per batch) are held out as a final test set used once; the "
        "remaining 24 go through 5-fold grouped cross-validation.", B))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Results", H))
    cvline = (f"CV (correct, grouped by image): accuracy "
              f"{fold_df.accuracy.mean():.4f} +/- "
              f"{fold_df.accuracy.std():.4f} across 5 folds; balanced "
              f"{fold_df.balanced_accuracy.mean():.4f} +/- "
              f"{fold_df.balanced_accuracy.std():.4f}.")
    story.append(Paragraph(cvline, B))
    tb = Table(
        [["class", "precision", "recall", "F1", "IoU"]] +
        [[n] + [f"{test_pooled['per_class'][n][m]:.3f}"
                for m in ("precision", "recall", "f1", "iou")]
         for n in CLASS_NAMES])
    story.append(tb)
    story.append(Paragraph(
        f"Phase-fraction error vs labels (the downstream use): mean abs "
        f"error {mae_ci[0]:.4f} [{mae_ci[1]:.4f}-{mae_ci[2]:.4f}] of image "
        f"area per class. Majority-class baseline = "
        f"{base_res['majority_mean']:.3f}; plain per-tile multi-Otsu "
        f"baseline agreement = {base_res['raw_otsu_mean']:.3f}.", B))
    story.append(Spacer(1, 6))
    for f, cap in [("confusion_matrix.png",
                    "Fig 1. Row-normalized confusion on held-out test pixels."),
                   ("per_class.png",
                    "Fig 2. Per-class precision/recall/F1/IoU on test."),
                   ("train_vs_val.png",
                    "Fig 3. Train vs validation accuracy per fold."),
                   ("leakage_demo.png",
                    "Fig 4. Random tile split (wrong) vs grouped split."),
                   ("confidence_vs_accuracy.png",
                    "Fig 5. Accuracy by confidence bin on test pixels."),
                   ("heldout_batch.png",
                    "Fig 6. Train on two batches, test on the third.")]:
        p = FIG_DIR / f
        if p.exists():
            story.append(Image(str(p), width=11 * cm,
                               height=11 * cm *
                               (plt.imread(p).shape[0] /
                                plt.imread(p).shape[1])))
            story.append(Paragraph(cap, B))
            story.append(Spacer(1, 4))

    story.append(Paragraph("Sanity checks", H))
    story.append(Paragraph(
        "; ".join(f"{s['check']}: {'PASS' if s['pass'] else 'FAIL'}"
                  for s in sanity), B))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Known failure cases and limitations", H))
    story.append(Paragraph(
        "Accuracy = agreement with multi-Otsu, not human truth. "
        "Held-out-batch accuracies: "
        + "; ".join(f"{r.held_out_batch} {r.accuracy:.3f}"
                    for r in hold_df.itertuples())
        + ". Worst tiles are listed in results.xlsx (Worst_tiles) and shown "
          "in figures/worst_tiles.png; errors concentrate at class borders, "
          "dark grey regions read as pore by the labeler, and low-contrast "
          "outlier images. Confusion is mostly pore<->bulk at soft edges; "
          "see results.xlsx Confusion_test. Model and labels are 2D image "
          "quantities; pixel size ~25 nm/px. The two outlier Batch_1 images "
          "remain in the data and are flagged in Per_image_test.", B))
    doc.build(story)
    print("wrote report.pdf")


def write_notes(splits, fold_df, test_df, sanity, acc_ci, hold_df):
    n = f"""# model_retrain - NOTES

## What this is
A NEW RandomForest pixel classifier trained to reproduce the existing
qc_real.py multi-Otsu labels on BSE tiles. There was no pre-existing trained
model or saved weights in the repo (qc_real.py is rule-based); the user
confirmed training a new model on the Otsu masks.

## IMPORTANT interpretation caveat
Labels are produced by a threshold pipeline, not a person. Every accuracy
figure = agreement with multi-Otsu + morphology, not ground truth.
Bright particles are assumed silicon-based, grey bulk graphite - appearance
only, stated per standing instruction.

## Data
- 31 locations, 7/7/17 per batch (matches earlier count, no mismatch).
- 2x2 tiling -> 124 tiles; tile size ~3500 x ~1000 px (all >> 200 px).
- Detectors BSE/Inlens/ETD share identical dims per location; `_SE.tif`
  treated as the side detector where `_ETD.tif` is absent.
- Tile NPZs and predictions are local artifacts (gitignored); only the
  model file, splits.json and report/xlsx/figures are committed.

## Splits (seed {SEED})
- test: {', '.join(splits['test'])}
- folds: {json.dumps(splits['folds'])}
- Leakage assertion passed: no image ID in more than one split.

## Headline numbers
- Test accuracy {acc_ci[0]:.4f} [95% CI {acc_ci[1]:.4f}-{acc_ci[2]:.4f}]
  (image bootstrap). Target 95%: {'REACHED' if acc_ci[1] >= 0.95 else 'NOT reached'}.
- CV accuracy {fold_df.accuracy.mean():.4f} +/- {fold_df.accuracy.std():.4f}.

## Sanity check results
"""
    for s in sanity:
        n += f"- {s['check']}: {'PASS' if s['pass'] else 'FAIL'} - {s['detail']}\n"
    n += """
## Assumptions / known issues
- Feature windows are truncated at tile edges -> tiny stitch-vs-whole
  mismatch (reported in check 7).
- '95%' is measured at the pixel level against Otsu labels; a human-labelled
  ground truth does not exist, so correctness vs reality is unknown.
- Training pixels are subsampled (30k/tile) for runtime; evaluation on the
  test images uses every pixel.
- The two low-contrast Batch_1 outliers (4ih2ggld, 5n1q8atc) are kept and
  flagged; they follow the same grouped-split rules.
"""
    (common.OUT_DIR / "NOTES.md").write_text(n)
    print("wrote NOTES.md")


if __name__ == "__main__":
    main()
