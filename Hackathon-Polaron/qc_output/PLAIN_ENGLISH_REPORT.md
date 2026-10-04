# What we found — in plain terms — SUPERSEDED

> **Status (acquisition analysis, 2026-10-04):** the REJECT/INVESTIGATE
> verdicts below were built on pore/particle *counts*, which we later
> showed follow the imaging session more than the batch itself. The
> current, session-corrected picture: **can't tell** whether the
> batches differ — the measurement system couldn't separate camera
> conditions from material. See `AGENTS.md` "Current state" and
> `paper/micro2dfn_paper.pdf`. Kept for provenance.

**The setup.** Suppliers deliver batches of battery electrode material. Before it goes into a cell, we check whether it still looks like the batch we already approved. We photograph each batch under a microscope and measure three things that matter: how much **silicon** it contains (the ingredient that stores energy but swells ~3x when charging), how much **empty space** it has (the breathing room silicon expands into), and how everything is **arranged** (clumps, flat pores, cracks).

**The method.** We measured the same ~40 properties on every photo, learned how much those numbers naturally wobble in the approved batch, then asked of each new batch: "anything here beyond what good material ever does?" Every measurement uses one fixed ruler, so no batch can re-normalise itself invisible.

**The verdicts.**
- **Batch_1: REJECT — quarantine it.** In 3 of 7 regions we photographed, the material carries clearly more silicon than anything in the approved batch, and the empty space those particles would need to expand into is nearly gone. The pattern matches "too much silicon dosed in" — likely an inconsistent mix at the supplier, since the rest of the batch looks normal. On charge, those pockets would swell against no room and crack. Ask the supplier what changed.
- **Batch_2: INVESTIGATE.** Two measurements moved beyond what good material does (si particle elongation and inlens crack area fraction), but they don't match any known defect pattern — possibly a change in imaging conditions rather than material. Verify before acting.

**What this does NOT tell you.** Photos can't show everything: we can't see inside the third dimension, and we can't predict exact battery lifespan. What we can say with numbers is whether the material changed, where, by how much, and which known manufacturing problem the change resembles.

Full technical detail: `qc_report.md` · all numbers: `polaron_qc_results.xlsx` and the CSV files in this folder.