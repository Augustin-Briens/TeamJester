"""Validation: do generated volumes reproduce the real 2-D statistics?

Patch-level holdout — generated (z,x) and (z,y) slices are compared
against real V x V crops on the statistics a microstructure model must
reproduce: phase fractions, two-point correlation S2(r) per phase, and
chord-length distributions.  Reconstructions are statistical
realisations — a pass means "consistent with the training statistics",
not "equal to the real 3-D structure" (unknowable from one view).
"""
from __future__ import annotations

import numpy as np

from . import config

PHASES = {"pore": 0, "bulk": 1, "bright": 2}


def two_point_corr(mask: np.ndarray, max_r: int = config.S2_MAX_R
                   ) -> np.ndarray:
    """Radial two-point correlation S2(r) of a boolean 2-D field via
    FFT autocorrelation (circular).  S2(r) = P(x and x+r in phase)."""
    f = mask.astype(np.float64)
    n = f.size
    ac = np.fft.ifft2(np.fft.fft2(f) *
                      np.conj(np.fft.fft2(f))).real / n
    ac = np.fft.fftshift(ac)
    c = np.array(ac.shape) // 2
    yy, xx = np.mgrid[:ac.shape[0], :ac.shape[1]]
    r = np.hypot(yy - c[0], xx - c[1]).astype(int)
    sums = np.bincount(r.ravel(), ac.ravel())
    cnt = np.bincount(r.ravel())
    prof = sums / np.maximum(cnt, 1)
    return prof[:max_r + 1]


def corr_length(mask: np.ndarray, um: float,
                max_r: int = config.S2_MAX_R) -> float:
    """r where normalised covariance (S2 - p^2)/(p - p^2) crosses 1/e —
    same definition as micro2dfn's corr_len markers."""
    p = float(mask.mean())
    if p <= 0 or p >= 1:
        return 0.0
    s2 = two_point_corr(mask, max_r)
    norm = (s2 - p * p) / max(p * (1 - p), 1e-12)
    below = np.where(norm < 1.0 / np.e)[0]
    return float(below[0]) * um if len(below) else float(max_r) * um


def chord_lengths(mask: np.ndarray, axis: int,
                  max_len: int = 128) -> np.ndarray:
    """Chord-length histogram (1..max_len px) along `axis`."""
    runs = mask if axis == 1 else mask.T
    d = np.diff(runs.astype(np.int8), axis=1, prepend=0, append=0)
    starts = np.argwhere(d == 1)
    stops = np.argwhere(d == -1)
    if not len(starts) or len(starts) != len(stops):
        return np.zeros(max_len)
    lengths = np.clip(stops[:, 1] - starts[:, 1], 1, max_len)
    hist = np.bincount(lengths, minlength=max_len + 1)[1:max_len + 1]
    return hist / max(hist.sum(), 1)


def generated_slices(vols: np.ndarray, n: int, seed: int = 0
                     ) -> np.ndarray:
    """n random z-containing slices from generated volumes -> (n,V,V)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = vols[rng.integers(0, len(vols))]
        a = rng.choice((1, 2))          # slice normal to y or x
        i = rng.integers(0, v.shape[a])
        out.append(v[:, i, :] if a == 1 else v[:, :, i])
    return np.stack(out)


def validate_batch(real_crops: np.ndarray, vols: np.ndarray,
                   voxel_um: float, n_fake: int = config.N_VAL_FAKE,
                   seed: int = 0) -> dict:
    """Compare held-out real crops vs generated slices.

    real_crops: (n,V,V) uint8 label images; vols: (m,V,V,V)."""
    fake = generated_slices(vols, n_fake, seed)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(real_crops),
                      min(len(real_crops), config.N_VAL_REAL),
                      replace=False)
    real = real_crops[pick]
    out = {"n_real": len(real), "n_fake": len(fake)}
    for name, ph in PHASES.items():
        rp = np.array([(r == ph).mean() for r in real])
        fp = np.array([(f == ph).mean() for f in fake])
        out[f"{name}_frac_real"] = float(rp.mean())
        out[f"{name}_frac_fake"] = float(fp.mean())
        out[f"{name}_frac_diff"] = float(fp.mean() - rp.mean())
        s2r = np.mean([two_point_corr(r == ph) for r in real], axis=0)
        s2f = np.mean([two_point_corr(f == ph) for f in fake], axis=0)
        out[f"{name}_s2_maxabsdiff"] = float(
            np.abs(s2r - s2f).max())
        cr = np.mean([corr_length(r == ph, voxel_um) for r in real])
        cf = np.mean([corr_length(f == ph, voxel_um) for f in fake])
        out[f"{name}_corr_len_real_um"] = float(cr)
        out[f"{name}_corr_len_fake_um"] = float(cf)
        for ax, tag in ((0, "z"), (1, "ip")):
            hr = np.mean([chord_lengths(r == ph, ax)
                          for r in real], axis=0)
            hf = np.mean([chord_lengths(f == ph, ax)
                          for f in fake], axis=0)
            k = min(len(hr), len(hf))
            out[f"{name}_chord_{tag}_l1"] = float(
                np.abs(hr[:k] - hf[:k]).sum())
    return out


def summary_verdict(v: dict) -> str:
    """Plain-language validation grade for the report."""
    fracs = [abs(v[f"{p}_frac_diff"]) for p in PHASES]
    s2 = [v[f"{p}_s2_maxabsdiff"] for p in PHASES]
    if max(fracs) < 0.02 and max(s2) < 0.05:
        return ("PASS — generated slices reproduce volume fractions and "
                "two-point correlations within tight tolerance")
    if max(fracs) < 0.05 and max(s2) < 0.10:
        return ("ACCEPTABLE — small statistical mismatches; transport "
                "metrics usable with caution")
    return ("POOR — generated volumes do not match the training "
            "statistics; treat transport metrics as illustrative only")
