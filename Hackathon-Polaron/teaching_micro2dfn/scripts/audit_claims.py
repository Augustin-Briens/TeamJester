#!/usr/bin/env python3
"""
audit_claims.py — re-verify the machine-checkable claims in
claim_evidence.csv against the current source files.

Writes:  claim_audit.csv  (per claim: expected, actual, status)
         claim_audit.md   (human summary; FAIL/STALE need action)

Statuses:
  PASS          value in the source file matches the claim
  STALE         source file exists but the value no longer matches —
                the deck/document text must be updated (or the claim
                is describing a different run than the one on disk)
  MISSING       source file/row the claim cites is gone
  MANUAL        claim is a synthesis/doc-level statement — cannot be
                machine-checked; reviewed by a human

Run:  .venv/bin/python teaching_micro2dfn/scripts/audit_claims.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.dirname(HERE)
ROOT = os.path.dirname(TEACH)

RESULTS = []


def report(cid, name, status, expected, actual, note=""):
    RESULTS.append(dict(claim_id=cid, name=name, status=status,
                        expected=str(expected), actual=str(actual),
                        note=note))
    mark = {"PASS": "PASS", "STALE": "STALE", "MISSING": "MISSING",
            "MANUAL": "MANUAL", "SKIP": "SKIP ",
            "WITHDRAWN": "WITHDRAWN"}[status]
    print(f"  [{mark}] {cid:5s} {name:58s} expected={expected} "
          f"actual={actual} {note}")


def num(cid, name, actual, expected, tol=0.02, note=""):
    """PASS if |actual-expected| <= tol*|expected|."""
    try:
        good = abs(float(actual) - float(expected)) <= tol * max(
            abs(float(expected)), 1e-12)
    except (TypeError, ValueError):
        good = False
    report(cid, name, "PASS" if good else "STALE", expected,
           round(float(actual), 6) if actual is not None else "n/a",
           note)
    return good


def text(cid, name, path, *needles):
    """PASS if all needles appear in the file text."""
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        report(cid, name, "MISSING", f"file {path}", "absent")
        return False
    t = open(p, encoding="utf8", errors="replace").read()
    miss = [n for n in needles if str(n) not in t]
    report(cid, name, "PASS" if not miss else "STALE",
           f"contains {needles}", f"missing {miss}" if miss else "ok")
    return not miss


def load_csv(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def load_json(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


print("auditing claims against current sources\n")

# ---------------------------------------------------------------- C01-03
man = load_csv("teaching_micro2dfn/dataset_manifest.csv")
if man is None:
    report("C01", "manifest exists", "MISSING", "34 rows", "no file")
else:
    num("C01", "FOV count", len(man), 34, 0)
    bc = man.batch.value_counts()
    num("C01", "B1 count", bc.get("Batch_1", 0), 7, 0)
    num("C01", "B2 count", bc.get("Batch_2", 0), 7, 0)
    num("C01", "B3 count", bc.get("Batch_3", 0), 17, 0)
    num("C01", "new count", bc.get("New_Images_Batch", 0), 3, 0)
    for col, want, cid in [("has_inlens", 34, "C02"),
                           ("has_etd", 30, "C02"), ("has_se", 4, "C02")]:
        if col in man.columns:
            num(cid, f"{col} count", int(man[col].sum()), want, 0)
        else:
            report(cid, col, "MISSING", "column", "absent")
    num("C03", "pixel pitch um", man.pixel_um.median(), 0.025, 0.01)

# ---------------------------------------------------------------- C05-08
rec = load_json("validated_comparison/recipe.json") or {}
num("C05", "t_pore", rec.get("t_pore"), 0.7108650803565979, 1e-6)
num("C05", "t_si", rec.get("t_si"), 1.3353633880615234, 1e-6)
num("C05", "t_core", rec.get("t_core"), 1.549543023109436, 1e-5)
num("C05", "ref n", rec.get("n_reference_images"), 17, 0)
prm = rec.get("params_um", {})
if prm:
    num("C07", "min bright um2", prm.get("min_bright_area_um2"),
        150 * 0.025**2, 0.05)
    num("C07", "min pore um2", prm.get("min_pore_area_um2"),
        20 * 0.025**2, 0.05)

# C06 — micro2dfn thresholds; tolerate BOTH the legacy pooled-31 file
# (thresholds.json fitted_on_n_images=31) and the new baseline-fitted
# recipe.json written by the refactored run_dfn.py.
mk_th = load_json("dfn_output/thresholds.json")
mk_rec = load_json("dfn_output/recipe.json")
if mk_rec:     # new-format frozen recipe exists
    num("C06", "mk t_pore (recipe)", mk_rec.get("t_pore"),
        0.7108652591705322, 1e-4,
        "baseline-fitted recipe (Batch_3, n=17)")
    num("C06", "mk t_si (recipe)", mk_rec.get("t_si"),
        1.335363745689392, 1e-4)
    report("C06", "recipe fit basis", "MANUAL",
           "fitted_on field", mk_rec.get("fitted_on"),
           "deck text 'pooled 31' is stale if fitted_on != pooled")
elif mk_th is not None:
    num("C06", "mk t_pore", mk_th.get("t_pore"), 0.6775720119476318,
        1e-4)
    num("C06", "mk t_si", mk_th.get("t_si"), 1.330653429031372, 1e-4)
    num("C06", "mk fit n", mk_th.get("fitted_on_n_images"), 31, 0)
else:
    report("C06", "micro2dfn thresholds", "MISSING",
           "thresholds.json or recipe.json", "neither found")

# ---------------------------------------------------------------- C09-13
pw = load_csv("validated_comparison/pairwise_comparisons.csv")
if pw is None:
    report("C09", "pairwise table", "MISSING", "present", "absent")
else:
    num("C09", "declared features", pw.feature.nunique(), 9, 0)
    num("C09", "pairs", pw.pair.nunique(), 3, 0)


def pwrow(feat, pair):
    r = pw[(pw.feature == feat) & (pw.pair == pair)]
    return r.iloc[0] if len(r) else None


for cid, feat, pair, d_want, p_want in [
        ("C10", "pores_per_mpx", "Batch_1-Batch_3", 34.4961,
         0.0004767),
        ("C11", "pores_per_mpx_noise_adj", "Batch_1-Batch_3", 18.7656,
         0.0018636),
        ("C12", "pores_per_mpx", "Batch_2-Batch_3", 19.4772, 0.01007),
        ("C12", "pores_per_mpx_noise_adj", "Batch_2-Batch_3", 4.1577,
         0.3528)]:
    r = pwrow(feat, pair)
    if r is None:
        report(cid, f"{feat} {pair}", "MISSING", "row", "absent")
    else:
        num(cid, f"{feat} {pair} diff", r.median_diff, d_want, 0.001)
        num(cid, f"{feat} {pair} p", r.p_exact, p_want, 0.01)

r = pwrow("si_candidate_frac", "Batch_1-Batch_3")
if r is not None:
    num("C13", "si MDD80", r.mdd_80pct, 0.0236, 0.05)
n_mdd = int(pw.mdd_80pct.notna().sum()) if pw is not None else 0
num("C48", "rows with MDD", n_mdd, len(pw), 0)

# ---------------------------------------------------------------- C14
pif = load_csv("validated_comparison/per_image_features.csv")
if pif is None:
    report("C14", "per-image features", "MISSING", "present", "absent")
else:
    for img, want in [("img_4ih2ggld", 1.77), ("img_5n1q8atc", 1.82),
                      ("img_f1vzngrs", 2.36)]:
        rr = pif[pif.image_id == img]
        num("C14", f"{img} contrast",
            rr.si_bulk_contrast.iloc[0] if len(rr) else None, want,
            0.03)
    b3 = pif[pif.batch == "Batch_3"]
    fl = pif[pif.image_id.isin(["img_4ih2ggld", "img_5n1q8atc",
                                "img_f1vzngrs"])]
    if "bright_frac_raw" in pif.columns:
        num("C41", "flagged bright min", fl.bright_frac_raw.min(),
            0.107, 0.01)
        num("C41", "flagged bright max", fl.bright_frac_raw.max(),
            0.125, 0.01)
        num("C41", "B3 bright median", b3.bright_frac_raw.median(),
            0.077, 0.02)
        num("C41", "B3 bright max", b3.bright_frac_raw.max(), 0.107,
            0.02)
        num("C44", "classifier gap max (pts)",
            100 * (fl.bright_frac_raw - fl.si_candidate_frac).max(),
            9.5, 0.1)

# ---------------------------------------------------------------- C16-20
ad = load_csv("reconcile_output/adjudication.csv")
if ad is None:
    report("C16", "adjudication table", "MISSING", "present", "absent")
else:
    t_all = " ".join(ad.astype(str).values.ravel())
    for cid, needles in [("C16", ("94%", "45%")),
                         ("C17", ("1.05", "0.82", "8%")),
                         ("C18", ("20.7", "11.9", "1.179")),
                         ("C19", ("1.20", "1.23")),
                         ("C45", ("13%",))]:
        miss = [n for n in needles if n not in t_all]
        report(cid, "adjudication " + "/".join(needles),
               "PASS" if not miss else "STALE", f"contains {needles}",
               f"missing {miss}" if miss else "ok")

vm = load_csv("reconcile_output/verdict_matrix.csv")
if vm is not None:
    num("C22", "verdict agreements", int(vm.agree.sum()),
        len(vm) - 1, 0)
    r = vm[vm.image_id == "img_f1vzngrs"]
    if len(r):
        num("C20", "f1vzngrs bright z", r.bright_frac_raw_z.iloc[0],
            2.4, 0.02)
        num("C20", "f1vzngrs contrast z", r.si_bulk_contrast_z.iloc[0],
            0.6, 0.1)
mc = load_csv("reconcile_output/metric_crosswalk.csv")
if mc is not None:
    col = "corr" if "corr" in mc.columns else mc.columns[-1]
    si = mc[mc.astype(str).apply(
        lambda r: r.str.contains("si_|classified", case=False,
                                 regex=True).any(), axis=1)]
    if len(si):
        num("C21", "classified-Si corr", float(si[col].iloc[0]), 0.27,
            0.2)

# ---------------------------------------------------------------- C23-26
qv = load_csv("qc_output/verdicts.csv")
if qv is not None:
    r1 = qv[qv.batch == "Batch_1"].verdict.iloc[0]
    r2 = qv[qv.batch == "Batch_2"].verdict.iloc[0]
    report("C23", "original verdicts",
           "PASS" if (r1 == "REJECT" and r2 == "INVESTIGATE")
           else "STALE", "REJECT/INVESTIGATE", f"{r1}/{r2}")
qr = load_csv("validated_comparison/qc_recheck.csv")
if qr is not None:
    t = qr.astype(str).values.ravel()
    ok = "INVESTIGATE" in " ".join(t)
    report("C24", "corrected recheck INVESTIGATE",
           "PASS" if ok else "STALE", "INVESTIGATE present",
           "found" if ok else "absent")
cc = load_csv("validated_comparison/calibration_counts.csv")
if cc is not None:
    num("C25", "exceedances", int(cc.n_exceedances.iloc[0]), 1, 0)
    num("C25", "exceedance rate", cc.exceedance_rate.iloc[0], 0.0588,
        0.01)
    num("C25", "CI lo", cc.exceedance_rate_ci_lo.iloc[0], 0.0105, 0.05)
    num("C25", "CI hi", cc.exceedance_rate_ci_hi.iloc[0], 0.2698, 0.05)
loo = load_csv("validated_comparison/loo_sweep.csv")
if loo is not None and pif is not None:
    d = loo[(loo.pair == "Batch_1-Batch_3")
            & (loo.feature == "pores_per_mpx")]
    num("C26", "LOO max p", d.p_exact.max(), 0.001605, 0.05)

# ---------------------------------------------------------------- C27-32
dsum = load_csv("dfn_output/dfn_indicator_summary.csv")
if dsum is not None:
    col = "n_converged" if "n_converged" in dsum.columns else None
    if col:
        num("C27", "runs converged", int(dsum[col].max()), 17, 0)
dd = load_csv("dfn_output/dfn_paired_differences.csv")
if dd is not None:
    for cid, ind, pair, want, below in [
            ("C28", "discharge_cap_last_Ah", "Batch_1-Batch_3",
             0.10929, True),
            ("C28", "discharge_cap_last_Ah", "Batch_2-Batch_3",
             0.150, True),
            ("C28", "discharge_cap_last_Ah", "Batch_1-Batch_2",
             -0.046588, True),
            ("C30", "si_stress_MPa", "Batch_1-Batch_3", -0.950478,
             False),
            ("C30", "si_stress_MPa", "Batch_2-Batch_3", -1.597265,
             False),
            ("C30", "si_stress_MPa", "Batch_1-Batch_2", 0.525519,
             False)]:
        r = dd[(dd.indicator == ind) & (dd.pair == pair)]
        if len(r):
            num(cid, f"{ind} {pair}", r.median_paired_diff.iloc[0],
                want, 0.02)
    num("C29", "assumption IQR",
        dd.assumption_only_spread_iqr.iloc[0], 1.1376, 0.02)

# ---------------------------------------------------------------- C33-39
text("C33", "reference cell values", "dfn_output/dfn_validation.md",
     "5.401", "5.415", "0.0115")
text("C34", "ablation", "dfn_output/dfn_validation.md", "70.63",
     "16.03")
text("C35", "constants audit", "dfn_output/dfn_validation.md",
     "2.7778e-07", "1e-12", "1e-15")
text("C36", "validation 8/8", "dfn_output/dfn_validation.md", "8/8")
text("C39", "tortuosity limit", "dfn_output/report.md", "span")
at = load_csv("dfn_output/assumption_table.csv")
if at is not None:
    t = " ".join(at.astype(str).values.ravel())
    for cid, needles in [("C37", ("75", "65-85",
                                 "measured gr_radius_eff_um",
                                 "35-90", "0.22-0.28", "0.15")),
                         ("C38", ("not a DFN input",))]:
        miss = [n for n in needles if n not in t]
        report(cid, "assumption table " + "/".join(needles),
               "PASS" if not miss else "STALE", f"contains {needles}",
               f"missing {miss}" if miss else "ok")

# ---------------------------------------------------------------- C40
asg = load_csv("new_image_assignment/assignments.csv")
if asg is not None:
    # all calls relabelled can't-tell: each photo's acquisition group
    # occurs in only one batch, so assignment can't separate
    # session-match from batch-match. Original call kept in
    # assigned_batch_original.
    for img, want_b in [("img_3e122cbj", "Batch_1"),
                        ("img_fn0mhxef", "Batch_2"),
                        ("img_xrv9xvzb", "Batch_3")]:
        r = asg[asg.image_id == img]
        if not len(r):
            report("C40", img, "MISSING", "can't tell", "no row")
            continue
        r = r.iloc[0]
        ok = (str(r.assigned_batch).startswith("cant_tell")
              and getattr(r, "assigned_batch_original", want_b) == want_b)
        report("C40", f"{img} -> can't tell (was {want_b})",
               "PASS" if ok else "STALE",
               f"cant_tell; original={want_b}",
               f"{r.assigned_batch} "
               f"(orig {getattr(r, 'assigned_batch_original', '?')})")

# ---------------------------------------------------------------- C42
of = load_csv("archive/orientation_features.csv")
report("C42", "orientation metrics (FFT direction test)",
       "WITHDRAWN", "—", "—",
       "fft_vh measures frame aspect ratio (noise +0.70 at 1034x3500); "
       "autocorrelation shows ~1.4x horizontal elongation. Archived.")

# ---------------------------------------------------------------- C46-50
for cid, name in [("C46", "DFN inputs identical -> identical "
                          "indicators (doc-level)"),
                  ("C47", "verdict semantics (doc-level)"),
                  ("C50", "anomaly exists, label calibration-dependent "
                          "(synthesis)")]:
    report(cid, name, "MANUAL", "—", "—",
           "synthesis claim; human-verified")

# ---------------------------------------------------------------- output
df = pd.DataFrame(RESULTS)
out_csv = os.path.join(TEACH, "claim_audit.csv")
df.to_csv(out_csv, index=False)

n_pass = (df.status == "PASS").sum()
n_stale = (df.status == "STALE").sum()
n_miss = (df.status == "MISSING").sum()
n_man = (df.status == "MANUAL").sum()
lines = ["# Claim audit — " + pd.Timestamp.now("UTC").strftime(
         "%Y-%m-%d %H:%M UTC"), "",
         f"PASS {n_pass} · STALE {n_stale} · MISSING {n_miss} · "
         f"MANUAL {n_man}", ""]
if n_stale or n_miss:
    lines += ["## Action needed — source no longer matches the claim",
              "",
              "| claim | expected | actual |", "|---|---|---|"]
    for _, r in df[df.status.isin(["STALE", "MISSING"])].iterrows():
        lines.append(f"| {r.claim_id} {r.name} | {r.expected} | "
                     f"{r.actual} |")
    lines += ["", "Update the deck text / claim_evidence.csv to match "
              "the current sources, then re-run sync."]
lines += ["", "## All checks", "", "| claim | check | status | "
          "expected | actual |", "|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append(f"| {r.claim_id} | {r.name} | {r.status} | "
                 f"{r.expected} | {r.actual} |")
open(os.path.join(TEACH, "claim_audit.md"), "w").write("\n".join(lines))
print(f"\nwrote {out_csv} and claim_audit.md — "
      f"{n_pass} pass / {n_stale} stale / {n_miss} missing")
sys.exit(1 if (n_stale or n_miss) else 0)
