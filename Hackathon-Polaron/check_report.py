"""Stale-number and archived-route check for the clean deliverables.

Fails (exit 1) if any forbidden stale value or archived route is
referenced in the generated paper or reports. Run after build_paper.py.
"""
from __future__ import annotations

import re
import sys

PAPER = "paper/micro2dfn_paper.tex"

# forbidden stale strings/values from superseded runs or archived claims
FORBIDDEN = [
    "4313", "1714",                     # old object counts
    "14.3", "81.8", "0.37", "0.39",     # old buggy DFN numbers
    "vertical texture is material",
    "fft_vh",
    "tau_fdm", "pore_spans",
    "9.5--10.0", "0.095 & 0.098",       # old marker table
    "9-point", "27 solves", "3 cycles",
    "1.555",                            # old t_core
    "more clustered than Batch",
    "excess silicon",
    "process variation, not formula",
    "artefact budget",
    "Si more clumped",
    "fig_si_positions", "fig_deficit_map",
    "immune to the confound",
    "every Batch\\_1 photo has",
    "calendering texture",
]

ARCHIVED_ROUTES = ["recon3d/", "orientation_features",
                   "_fdm_tortuosity"]


def main() -> int:
    tex = open(PAPER).read()
    bad = []
    for s in FORBIDDEN + ARCHIVED_ROUTES:
        if s in tex:
            bad.append(s)
    # hand-typed-number guard: key values must match report_numbers
    import json
    n = json.load(open("report_numbers.json"))
    med = n["fractions"]["si_candidate_frac"]
    for b, v in med.items():
        pct = f"{v['med'] * 100:.1f}"
        if pct not in tex:
            bad.append(f"missing Si frac {b}={pct}")
    pf = n["fractions"]["pore_frac"]
    for b, v in pf.items():
        pct = f"{v['med'] * 100:.1f}"
        if pct not in tex:
            bad.append(f"missing pore frac {b}={pct}")
    if bad:
        print("STALE/FORBIDDEN content found:")
        for s in bad:
            print("  -", s)
        return 1
    print("check_report: clean — no stale numbers or archived routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
