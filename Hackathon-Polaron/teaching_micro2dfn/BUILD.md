# Rebuild instructions

## The one command (recommended)

```bash
cd /Users/raduvadanici/Downloads/Hackathon-Polaron
.venv/bin/python teaching_micro2dfn/scripts/sync.py
```

`sync.py` is the self-updating entry point: it fingerprints all inputs,
rebuilds only changed stages, refreshes `version_manifest.txt`, audits
every claim (`claim_audit.md`), and logs to `sync_log.md`. Use
`--check` to audit without building, `--force` to rebuild everything.

## Manual stage chain (what sync runs)

```bash
cd /Users/raduvadanici/Downloads/Hackathon-Polaron

# 1. dataset manifest (reads TIFFs, records md5 + pixel pitch)
.venv/bin/python teaching_micro2dfn/scripts/make_manifest.py

# 2. all figures + annotated crops (~4 min; runs two real PyBaMM
#    solves for fig_trace/fig_sweepbands-style assets and recomputes
#    the permutation null as a self-check)
.venv/bin/python teaching_micro2dfn/scripts/build_figures.py

# 3. the deck
.venv/bin/python teaching_micro2dfn/scripts/build_deck.py

# 4. the matching PDF (LibreOffice)
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
  --convert-to pdf --outdir teaching_micro2dfn \
  teaching_micro2dfn/micro2dfn_masterclass.pptx

# 5. render pages for visual QA
.venv/bin/python - <<'EOF'
import fitz
doc = fitz.open("teaching_micro2dfn/micro2dfn_masterclass.pdf")
for i, page in enumerate(doc):
    page.get_pixmap(dpi=110).save(
        f"teaching_micro2dfn/rendered/slide_{i+1:02d}.png")
print("rendered", len(doc))
EOF
```

## Inputs the build reads (all already in the repo)

- `Batch_1/ Batch_2/ Batch_3/ New_Images_Batch/` — TIFFs
- `validated_comparison/` — recipe.json, pairwise_comparisons.csv,
  per_image_features.csv, qc_recheck.csv, calibration_counts.csv,
  loo_sweep.csv, checks.csv, START_HERE.md, panels/
- `dfn_output/` — report.md, dfn_paired_differences.csv,
  dfn_indicator_summary.csv, assumption_table.csv,
  dfn_input_bands.csv, dfn_validation.md, thresholds.json
- `reconcile_output/` — adjudication.csv, metric_crosswalk.csv,
  verdict_matrix.csv
- `new_image_assignment/` — assignments.csv, feature_table.csv
- `qc_output/` — verdicts.csv, z_scores.csv
- `orientation_features.csv`
- `micro2dfn/` pipeline modules (for threshold/object code paths)

## Environment

Pinned in `version_manifest.txt` (Python 3.13.5; NumPy 2.5.3;
pandas 3.0.6; SciPy 1.18.1; scikit-image 0.26.0; matplotlib 3.11.2;
PyBaMM 26.9.0.0; PyMuPDF 1.28.2; python-pptx).

## Self-checks inside the build

- `build_figures.py` recomputes the exact permutation p for
  pores_per_mpx B1–B3 and asserts it equals the saved 0.000477.
- Every figure `note()` writes one row to `annotations.csv` recording
  what was marked and by what rule.
- `build_deck.py` fails loudly if any figure it references is missing.
- `audit_claims.py` (run by sync) re-verifies ~70 file-checkable
  claims; STALE/MISSING means a statement no longer matches the
  sources — exit code 1, details in `claim_audit.md`.
