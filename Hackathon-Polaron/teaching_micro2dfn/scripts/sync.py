#!/usr/bin/env python3
"""
sync.py — keep teaching_micro2dfn current with the repository.

What it does, in order:

  1. FINGERPRINTS every input this pack depends on (source outputs,
     image files, build scripts). Compares to build_state.json.
  2. REBUILDS only what the changes touch:
       image set changed        -> make_manifest.py
       sources/scripts changed  -> build_figures.py
       anything above changed   -> build_deck.py + PDF + rendered/
  3. AUDITS every machine-checkable claim in claim_evidence.csv
     against the current sources (audit_claims.py). STALE or MISSING
     claims are reported loudly and make the exit code non-zero —
     they mean a document/deck statement no longer matches what the
     files on disk say.
  4. REGENERATES version_manifest.txt and appends a sync_log.md entry.

Usage:
  .venv/bin/python teaching_micro2dfn/scripts/sync.py          # build
  .venv/bin/python teaching_micro2dfn/scripts/sync.py --check  # report
  .venv/bin/python teaching_micro2dfn/scripts/sync.py --force  # rebuild all

If the upstream pipeline is re-run (e.g. `run_dfn.py` after its
baseline-recipe refactor), run sync afterwards: figures, deck, PDF and
the claim audit all refresh; any claim the new outputs contradict is
listed as STALE for a human to reword.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.dirname(HERE)
ROOT = os.path.dirname(TEACH)
STATE = os.path.join(TEACH, "build_state.json")
LOG = os.path.join(TEACH, "sync_log.md")
PY = os.path.join(ROOT, ".venv", "bin", "python")
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

BATCH_DIRS = ["Batch_1", "Batch_2", "Batch_3", "New_Images_Batch"]
SOURCE_FILES = [
    "validated_comparison/recipe.json",
    "validated_comparison/pairwise_comparisons.csv",
    "validated_comparison/per_image_features.csv",
    "validated_comparison/qc_recheck.csv",
    "validated_comparison/calibration_counts.csv",
    "validated_comparison/loo_sweep.csv",
    "validated_comparison/robustness.csv",
    "validated_comparison/checks.csv",
    "validated_comparison/START_HERE.md",
    "validated_comparison/TECHNICAL_APPENDIX.md",
    "dfn_output/report.md",
    "dfn_output/dfn_results.csv",
    "dfn_output/dfn_paired_differences.csv",
    "dfn_output/dfn_indicator_summary.csv",
    "dfn_output/dfn_validation.md",
    "dfn_output/assumption_table.csv",
    "dfn_output/dfn_input_bands.csv",
    "dfn_output/thresholds.json",
    "dfn_output/recipe.json",
    "dfn_output/run_manifest.json",
    "reconcile_output/adjudication.csv",
    "reconcile_output/metric_crosswalk.csv",
    "reconcile_output/verdict_matrix.csv",
    "new_image_assignment/assignments.csv",
    "new_image_assignment/feature_table.csv",
    "new_image_assignment/assignment_report.md",
    "qc_output/verdicts.csv",
    "qc_output/z_scores.csv",
    "qc_output/scorecard.csv",
    "qc_output/consequences.csv",
    "AGENTS.md",
]
PACK_INPUTS = [
    "teaching_micro2dfn/metric_dictionary.csv",
    "teaching_micro2dfn/claim_evidence.csv",
    "teaching_micro2dfn/dataset_manifest.csv",
]
SCRIPT_FILES = [
    "teaching_micro2dfn/scripts/make_manifest.py",
    "teaching_micro2dfn/scripts/figlib.py",
    "teaching_micro2dfn/scripts/build_figures.py",
    "teaching_micro2dfn/scripts/build_deck.py",
    "teaching_micro2dfn/scripts/audit_claims.py",
]
# pipeline code changes can alter interpretation of outputs
CODE_FILES = ["run_dfn.py", "run_qc.py", "run_reconcile.py",
              "assign_new_images.py"]


def fp_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(rel):
    """md5 for small files; size+mtime for large image files."""
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    if st.st_size > 4_000_000:
        return f"s{st.st_size}m{st.st_mtime_ns}"
    return fp_md5(p)


def collect_fingerprints():
    fps = {}
    for b in BATCH_DIRS:
        d = os.path.join(ROOT, b)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".tif", ".tiff")):
                v = fingerprint(os.path.join(b, f))
                if v:
                    fps[f"img:{b}/{f}"] = v
    for rel in SOURCE_FILES + PACK_INPUTS + SCRIPT_FILES + CODE_FILES:
        v = fingerprint(rel)
        if v:
            fps[rel] = v
    return fps


def run(cmd, label):
    print(f"\n--- {label} ---")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
    for line in tail:
        print("   ", line)
    return r.returncode == 0


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"fingerprints": {}, "last_sync_utc": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift only, do not rebuild")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    state = load_state()
    old = state["fingerprints"]
    cur = collect_fingerprints()

    changed = sorted(k for k in cur if old.get(k) != cur[k])
    removed = sorted(k for k in old if k not in cur)
    img_changed = [k for k in changed if k.startswith("img:")]
    src_changed = [k for k in changed if not k.startswith("img:")]

    print(f"fingerprinted {len(cur)} inputs")
    if not changed and not removed and not args.force:
        print("no input changes since last build")
        drift = False
    else:
        drift = True
        print(f"changed inputs: {len(changed)} "
              f"({len(img_changed)} images, {len(src_changed)} files)"
              + (f" · removed: {len(removed)}" if removed else ""))
        for k in (changed[:12] + (["..."] if len(changed) > 12 else [])):
            print("   changed:", k)
        for k in removed[:6]:
            print("   removed:", k)

    if args.check:
        a = subprocess.run([PY, os.path.join(
            HERE, "audit_claims.py")], cwd=ROOT)
        print("\n(check-only) audit exit:", a.returncode)
        sys.exit(a.returncode)

    stages = []
    if args.force or img_changed or any(
            "make_manifest" in k or "dataset_manifest" in k
            for k in src_changed):
        stages.append(("manifest", [PY, os.path.join(
            HERE, "make_manifest.py")]))
    if args.force or drift:
        stages.append(("figures", [PY, os.path.join(
            HERE, "build_figures.py")]))
        stages.append(("deck", [PY, os.path.join(
            HERE, "build_deck.py")]))

    did = []
    ok = True
    for label, cmd in stages:
        if not run(cmd, label):
            print(f"!! {label} FAILED — stopping before downstream")
            ok = False
            break
        did.append(label)
        if label == "deck":
            ok = run([SOFFICE, "--headless", "--convert-to", "pdf",
                      "--outdir", TEACH,
                      os.path.join(TEACH, "micro2dfn_masterclass.pptx")],
                     "pdf export")
            ok = ok and run([PY, "-c", (
                "import fitz,os;"
                "d=fitz.open(os.path.join('%s',"
                "'micro2dfn_masterclass.pdf'));"
                "[p.get_pixmap(dpi=110).save(os.path.join('%s',"
                "'rendered','slide_%%02d.png'%%(i+1)))"
                " for i,p in enumerate(d)];"
                "print('rendered',len(d))") % (TEACH, TEACH)],
                "render slides")
            did += ["pdf", "render"]

    # version manifest refresh (cheap, always)
    run([PY, "-c", VERSION_SNIPPET], "version manifest")

    # state + log
    state["fingerprints"] = cur
    state["last_sync_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())
    state["built_stages"] = did
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1)

    audit_rc = subprocess.run([PY, os.path.join(
        HERE, "audit_claims.py")], cwd=ROOT).returncode

    entry = [
        f"## {state['last_sync_utc']}",
        f"- rebuilt: {', '.join(did) if did else 'nothing (no changes)'}",
        f"- changed inputs: {len(changed)} "
        f"({len(img_changed)} images, {len(src_changed)} files)",
        "- claim audit: " + ("CLEAN" if audit_rc == 0
           else "STALE/MISSING claims — see claim_audit.md"), ""]
    with open(LOG, "a") as f:
        f.write("\n".join(entry))

    print("\n=== sync complete ===")
    print("rebuilt:", ", ".join(did) if did else "nothing")
    print("claim audit:", "clean" if audit_rc == 0
          else "STALE/MISSING — see claim_audit.md")
    sys.exit(0 if (ok and audit_rc == 0) else 1)


VERSION_SNIPPET = r'''
import json, sys, importlib.metadata as im, datetime, os
ROOT = os.getcwd()
out = os.path.join("teaching_micro2dfn", "version_manifest.txt")
w = open(out, "w")
print("micro2dfn teaching pack - version manifest", file=w)
print("generated:", datetime.datetime.now().isoformat(timespec="seconds"),
      file=w)
print("\npython:", sys.version.split()[0], file=w)
for m in ["numpy","pandas","scipy","scikit-image","matplotlib",
          "tifffile","PyMuPDF","pybamm","python-pptx","Pillow"]:
    try: print(f"{m}: {im.version(m)}", file=w)
    except Exception: pass
r = json.load(open("validated_comparison/recipe.json"))
print("\ncomparison recipe fitted_on:", r.get("fitted_on"),
      "| seed:", r.get("seed"), file=w)
print("thresholds:", {k: r[k] for k in ("t_pore","t_si","t_core")},
      file=w)
for p in ("dfn_output/recipe.json", "dfn_output/thresholds.json"):
    if os.path.exists(p):
        t = json.load(open(p))
        print(f"{p}:", {k: t.get(k) for k in
             ("t_pore","t_si","t_core","fitted_on",
              "fitted_on_n_images") if k in t}, file=w)
st = os.path.join("teaching_micro2dfn", "build_state.json")
if os.path.exists(st):
    s = json.load(open(st))
    print("\nbuild_state last_sync:", s.get("last_sync_utc"), file=w)
    print("fingerprinted inputs:", len(s.get("fingerprints", {})),
          file=w)
w.close()
'''

if __name__ == "__main__":
    main()
