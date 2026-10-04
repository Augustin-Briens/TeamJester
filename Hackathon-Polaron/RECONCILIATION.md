# Reconciliation — polaron_qc × micro2dfn

> **Status (acquisition analysis, 2026-10-04):** the adjudication below
> ("likely fine silicon") leaned on count/texture evidence that is
> session-organised; the REJECT it discusses predates that analysis.
> What survives: both systems detect the *same* Batch_1 bright fields,
> and the uncertain-bright label is the honest bound between the two
> interpretations. Current verdict on the material question: **can't
> tell**. See `AGENTS.md` "Current state".

*Generated from `reconcile_output/` (run `run_reconcile.py` to
reproduce; all numbers below are from the current run.)*

## The question

Two independent pipelines analyse the same 31 SEM images:

- **polaron_qc** — frozen *baseline-fitted* thresholds, counts **all**
  bright pixels as silicon → Batch_1 regions img_4ih2ggld / img_5n1q8atc
  / img_f1vzngrs read **10.7–12.5 % Si** → verdict **REJECT (E1,
  "excessive silicon")**.
- **micro2dfn** — pooled-all-images thresholds + an object classifier
  that splits `si_particle` from ambiguous `bright_fine` → classifies
  most of the extra bright material in the same regions as
  `bright_fine` → `si_frac_total ≈ 6 %` everywhere → **no silicon
  anomaly**, only low-contrast "problem photo" flags + weaker
  clustering/connectivity differences.

Same images, same segmentation recipe — opposite interpretations of the
same bright material. Which is right?

## Answer: the extra material is real silicon

**5 of 6 adjudication tests support fine silicon; the sixth rules out
acquisition drift.** Conclusion: **LIKELY FINE SILICON** — with the
low image contrast on two of the three regions kept as a documented
secondary caveat.

### Evidence table

| Test | Statistic | Result | Direction |
|---|---|---|---|
| T1 interior rescaling | Share of `bright_fine` objects passing the particle cut after per-image contrast correction | **94 %** on flagged regions vs 45 % on baseline | fine-Si |
| T2 object morphology | Solidity / size / border share vs particle controls | solidity 0.82 (Si: 0.88), d≈1.05 µm, only 8 % at border, n=767 | fine-Si |
| T3 spatial independence | Adjacency to particles + shape of adjacent objects | adjacent objects are **compact discs** (solidity 0.84), not crescent rims; bf objects self-cluster (NN spacing 0.43× uniform) | fine-Si **clusters** |
| T4 floating-Otsu | Per-image Otsu bright fraction on flagged images | floats t_si **down** to 1.18 → counts **20.7 %** bright vs 11.9 % frozen — even *more* material | fine-Si |
| T5 InLens/ETD texture | Gradient energy inside bf regions vs Si vs bulk | **bf texture ≥ Si texture** on both channels (ratio 1.20 ETD, 1.23 InLens) — real topography, not a smooth smear | fine-Si |
| T6 acquisition signature | Raw median & bright-tail shifts | whole-image median **z = 0.0**; only the bright tail moved (z = 1.8) | material change |

### The decisive row in the verdict matrix

`img_f1vzngrs` — one of the three flagged regions — has **normal image
contrast** (Si:bulk contrast z = +0.6) yet still shows elevated bright
material (z = +2.4). The low-contrast anomaly affects only the other
two flagged regions (contrast z = −4.8, −4.2). So the extra bright
material exists **even where the contrast question does not apply** —
it cannot be wholly a contrast artefact.

### What the two flagged images' dimming means

`img_4ih2ggld` / `img_5n1q8atc` are genuinely lower-contrast
(Si interior ≈ 1.66 vs 1.88 baseline, contrast ratio 1.85 vs 2.39).
Two readings remain open and both are bad news for the batch:

- **Acquisition drift** on those two regions — then Si is even higher
  than the frozen ruler reports (a per-image refit sees 20.7 %), *and*
  the imaging itself was inconsistent across the batch.
- **Material difference in the silicon** (finer, oxidised, or different
  grade scatters fewer electrons) — which is a *type* of composition
  change, not an artefact.

Either way the batch is non-uniform and the REJECT stands.

## Where the systems agree (method cross-validation)

Per-image correlation on shared metrics (n = 31 images):

| Metric | r | Comment |
|---|---|---|
| Raw bright material | **0.92** | both measure the same bright pixels |
| Porosity | **1.00** | identical |
| Pore anisotropy / flake alignment | **0.96 / 0.95** | same physics |
| Si→pore distance | 0.85 | consistent |
| Vertical cracks | 0.77 | consistent |
| Si–pore contact | 0.63 | proximity definition, consistent |
| Swelling budget | 0.50 | diluted by the Si disagreement |
| **Classified silicon** | **0.27** | the bright_fine divergence |
| Si clustering | 0.32 | different splitting conventions |
| Pore fragmentation | −0.23 | related but different concepts |

Two independent codebases converging on pores, orientation, transport
and distance at r≈0.8–1.0 validates the measurement recipe. The single
structural disagreement is *exactly* the classifier decision
bright_frac (0.92 raw agreement) → si_frac_total (0.27 after
classification).

## Why the systems diverge — methodological diff

| Choice | polaron_qc | micro2dfn | Consequence |
|---|---|---|---|
| Threshold fit pool | baseline only | pooled all images | mk's t_si 1.40 vs qc 1.34 — incoming data pulls its own ruler |
| Bright-phase identity | all bright = Si | object classifier: interior ≥ t_core + solidity | mk demotes dim-interior objects; low-contrast regions lose counted Si |
| Aggregation | per-region robust-z + fingerprints | batch-median bootstrap CIs | mk's batch median dilutes B1's 3-of-7 anomaly |
| Surface channel | InLens cracks wired in | not used | B2's InLens shift invisible to mk |
| Consequence layer | cited proxy bands (anomalous-region scope) | PyBaMM DFN sweep (batch-median inputs) | DFN inputs identical → identical indicators; no discrimination yet |

## Verdict agreement

13 / 14 images agree on anomaly status. The single disagreement is
`img_f1vzngrs`, which QC flags (E1, 2.42 > q90 = 1.90) and micro2dfn
nearly flags itself (`bright_frac_raw` z = +2.39 vs its own >2.5 rule).

## Convergence actions taken

- `polaron_qc` report: E1-driver batches now carry a **contrast
  caveat** block referencing this adjudication (fires when driver
  images' Si:bulk contrast z < −2).
- `run_dfn.py --baseline Batch_3`: optional baseline-only threshold
  fit (QC-metrology convention); pooled fit remains default for
  exploratory use; `fit_mode` recorded in `thresholds.json`.
- Deferred (documented, not built): per-region DFN sweeps — run once
  the adjudication is externally confirmed; the batch-median DFN
  inputs currently erase the very heterogeneity the verdict is about.

## Reproduce

```bash
.venv/bin/python run_reconcile.py
# -> reconcile_output/{metric_crosswalk,adjudication,verdict_matrix}.csv
# -> reconcile_output/figs/*.png
```
