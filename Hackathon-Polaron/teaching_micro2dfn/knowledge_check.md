# Knowledge check — 8 questions (answers below)

1. A floating (per-image) threshold finds **more** bright material on
   the disputed Batch_1 photos than the frozen one (20.7% vs 11.9%).
   Does that make the frozen threshold wrong?

2. Batch_1's pore-count excess survives noise adjustment (p = 0.002)
   while Batch_2's raw shift does not. What does the noise-adjusted
   result actually establish about Batch_1 — and about Batch_2?

3. The two pipelines correlate at r ≈ 0.8–1.0 on every shared metric
   except classified silicon (r = 0.27). Where does the disagreement
   live?

4. The DFN model reports Batch_2 vs Batch_3 cycle-5 capacity
   +0.150 Ah. Why is this **not** evidence that Batch_2 cells would
   perform better?

5. "No clear difference was detected for Si fraction." Why is that
   not a claim of equivalence? What number bounds it?

6. Why is the image — not the pixel, and not the particle — the
   independent statistical unit?

7. The adjudication concludes "likely fine silicon" for the disputed
   material. Why is that a *supported interpretation* and not a
   conclusion?

8. The original QC run reported "zero baseline false alarms". What
   did corrected calibration measure instead, and what changed as a
   consequence?

---

## Answers

1. **No — it shows why a floating ruler can't be the measuring
   instrument.** A threshold refit per image re-centres on whatever
   the image contains; on an image with genuinely more bright
   material it moves the cut up and still finds more (and on other
   anomalies it could absorb the shift). Direction of bias is
   unpredictable. The frozen ruler is not "right" in an absolute
   sense — it is *the same for every image*, which is what makes
   images comparable. The floating value stays useful as a
   diagnostic (it told us the frozen ruler isn't over-counting on
   the flagged photos).

2. **Batch_1: a uniform, batch-level shift in resolved pore count
   that is not explained by image noise** — all 7 images sit above
   the Batch_3 median and the result survives leave-one-out
   (p ≤ 0.0016). **Batch_2: its raw-count excess was mostly a
   photography effect** (noise correlated with count at ~0.91);
   after removing the noise-predicted part, +4.2 with p = 0.35 — no
   clear difference at this n, *with an MDD* — not equivalence.

3. **In the object classifier, not in the pixels.** Both pipelines
   see the same segmentation; they disagree only on whether
   borderline bright objects are named si_particle or ambiguous
   material. That single naming decision is r = 0.27; everything
   else — porosity, anisotropy, raw bright fraction, alignment —
   agrees at r ≥ 0.8. The "two answers" are one measurement plus one
   labelling disagreement.

4. Three reasons: (a) it's a **model output**, not a measurement —
   no cell was tested; (b) +0.150 Ah is **below the 0.25 Ah
   meaningful-difference threshold declared before looking**; (c)
   the **assumption-only spread (~1.16 Ah IQR) dwarfs it** — the
   same sweep that produced the difference could hide it entirely.
   It is a conditional indicator, useful for ranking hypotheses,
   not a performance claim.

5. Because n is small (7 vs 17). Every non-significant row carries a
   **minimum detectable difference** (MDD, e.g. ~0.024 fraction for
   B1–B3 Si candidate share): real differences *smaller* than the
   MDD would not have been found. Honest statement: "no difference
   detectable at MDD ≈ X under this recipe."

6. Pixels and particles inside one image are correlated — they
   share a specimen, a polish, an imaging session. Treating them as
   independent would inflate n by ~10⁶ and make any trivial shift
   "significant". The 34 images are the real samples; per-image
   summaries are compared, and permutation tests shuffle *image*
   labels.

7. Because the six tests interrogate **pixels and morphology**
   (contrast-corrected classification, shape, neighbourhood,
   threshold sensitivity, texture, distribution) — all consistent
   with fine/dimmer silicon and inconsistent with pure imaging
   artefact — but none measures **chemistry**. Brightness in BSE
   tracks mean atomic number, not species. Only EDS (elemental
   spectroscopy) or equivalent closes it. Hence "likely", flagged
   open.

8. Corrected calibration (reference-only weights, held-out T², LOO)
   measured **1 of 17 reference images exceeding the flag threshold
   ≈ 5.9% [CI 1.0–26.9%]** — the "zero" claim was an artefact of
   fitting calibration on data that had seen the test batches.
   Consequence: Batch_1's automatic REJECT was revised to
   **INVESTIGATE** — the anomaly stands, the severity label was
   calibration-dependent.
