"""3-D steady-state diffusion through a voxel pore network.

Same physics as micro2dfn's 2-D `_fdm_tortuosity` (and porespy's
tortuosity_fd / TauFactor):  Laplace solve on the conductive phase with
c=1 on the source face, c=0 on the sink face, no-flux elsewhere.

D_eff_rel = total flux * L / A        (D0 = 1)
F (formation factor) = 1 / D_eff_rel
tau = F * phi = phi / D_eff_rel

Solves on the SPANNING cluster only — dead-end and isolated pores carry
no steady-state flux, so the result is identical to solving on the full
pore phase but much faster.  Isolated pores still count in phi.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from . import config

_CROSS = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                   [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=int)


def _faces(idx: np.ndarray, axis: int, n: int):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis], sl1[axis] = 0, n - 1
    return idx[tuple(sl0)], idx[tuple(sl1)]


def _pairs(idx: np.ndarray, axis: int):
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis], sl_b[axis] = slice(0, -1), slice(1, None)
    a, b = idx[tuple(sl_a)], idx[tuple(sl_b)]
    m = (a >= 0) & (b >= 0)
    return a[m], b[m]


def tortuosity(vol: np.ndarray, phase: int = 0, axis: int = 0,
               tol: float = config.CG_TOL,
               maxiter: int = config.CG_MAXITER,
               return_field: bool = False) -> dict:
    """Steady-state diffusion along `axis` through `phase` voxels.

    Returns tau, d_eff, phi, spanning/percolating stats and solver
    diagnostics.  axis=0 is the through-plane direction (volume
    convention z = electrode depth).
    """
    cond = vol == phase
    n = vol.shape[axis]
    phi = float(cond.mean())
    res = dict(axis=axis, phi=phi, tau=np.inf, d_eff=0.0,
               formation_factor=np.inf, spans=False,
               spanning_pore_frac=0.0, source_connected_frac=0.0,
               n_unknowns=0, n_iter=0, converged=False)
    if cond.sum() < 8:
        return res

    lab, _ = label(cond, structure=_CROSS)
    f0, f1 = _faces(lab, axis, n)
    src_ids = set(np.unique(f0)) - {0}
    sink_ids = set(np.unique(f1)) - {0}
    span_ids = src_ids & sink_ids
    src_mask = np.isin(lab, list(src_ids)) if src_ids else \
        np.zeros_like(cond)
    res["source_connected_frac"] = float(
        src_mask.sum() / cond.sum())
    if not span_ids:
        return res
    cluster = np.isin(lab, list(span_ids))
    res["spans"] = True
    res["spanning_pore_frac"] = float(cluster.sum() / cond.sum())

    idx = -np.ones(vol.shape, np.int64)
    idx[cluster] = np.arange(cluster.sum())
    nvar = int(cluster.sum())
    res["n_unknowns"] = nvar

    diag = np.zeros(nvar)
    rhs = np.zeros(nvar)
    # boundary links: voxels on the first/last plane along `axis`
    b0, b1 = _faces(idx, axis, n)
    b0 = b0[b0 >= 0]
    b1 = b1[b1 >= 0]
    diag[b0] += 1.0
    rhs[b0] += 1.0            # link to c = 1 source plane
    diag[b1] += 1.0           # link to c = 0 sink plane
    e_i, e_j = [], []
    for a in range(3):
        i, j = _pairs(idx, a)
        e_i += [i, j]
        e_j += [j, i]
        np.add.at(diag, i, 1.0)
        np.add.at(diag, j, 1.0)
    ei = np.concatenate(e_i) if e_i else np.array([], np.int64)
    ej = np.concatenate(e_j) if e_j else np.array([], np.int64)
    ii = np.concatenate([ei, np.arange(nvar)])
    jj = np.concatenate([ej, np.arange(nvar)])
    vv = np.concatenate([-np.ones(len(ei)), diag])
    A = csr_matrix((vv, (ii, jj)), shape=(nvar, nvar))
    M = csr_matrix((1.0 / diag, (range(nvar), range(nvar))),
                   shape=(nvar, nvar))
    it = [0]
    c, info = cg(A, rhs, rtol=tol, maxiter=maxiter, M=M,
                 callback=lambda xk: it.__setitem__(0, it[0] + 1))
    res["n_iter"] = it[0] if info >= 0 else int(info)
    res["converged"] = bool(info == 0)
    if info < 0:
        return res

    flux_out = float(c[b1].sum())          # sum of c at sink links
    flux_in = float((1.0 - c[b0]).sum())   # sum of (1-c) at source
    res["flux_balance"] = float(
        abs(flux_in - flux_out) / max(abs(flux_in), 1e-12))
    flux = flux_out
    area = int(np.prod([vol.shape[a] for a in range(3) if a != axis]))
    d_eff = flux * n / area
    res["d_eff"] = float(d_eff)
    if d_eff > 0:
        res["formation_factor"] = 1.0 / d_eff
        res["tau"] = phi / d_eff
    if return_field:
        field = np.full(vol.shape, np.nan, np.float32)
        field[cluster] = c.astype(np.float32)
        res["field"] = field
        res["cluster"] = cluster
    return res


def accessible_fraction(vol: np.ndarray, axis: int = 0,
                        pore_phase: int = 0, target_phase: int = 2,
                        contact: int = config.CONTACT_VOX) -> dict:
    """Share of `target_phase` voxels within `contact` vox of the
    spanning pore cluster — the 3-D version of the 2-D Si-pore
    proximity metric (electrolyte-reachable material)."""
    pore = vol == pore_phase
    lab, _ = label(pore, structure=_CROSS)
    n = vol.shape[axis]
    f0, f1 = _faces(lab, axis, n)
    span = set(np.unique(f0)) - {0}
    span &= set(np.unique(f1)) - {0}
    tgt = vol == target_phase
    if not span or not tgt.any():
        return dict(accessible_frac=np.nan, target_frac=float(tgt.mean()),
                    spans=bool(span))
    cl = np.isin(lab, list(span))
    reach = binary_dilation(cl, structure=_CROSS, iterations=contact)
    return dict(accessible_frac=float((reach & tgt).sum() / tgt.sum()),
                target_frac=float(tgt.mean()), spans=True)


def volume_metrics(vol: np.ndarray, voxel_nm: float = config.VOXEL_NM
                   ) -> dict:
    """All per-realization metrics for one label volume."""
    m = dict(pore_frac=float((vol == 0).mean()),
             bright_frac=float((vol == 2).mean()))
    tp = tortuosity(vol, phase=0, axis=0)
    ip = tortuosity(vol, phase=0, axis=1)
    m.update(tau_tp=tp["tau"], d_eff_tp=tp["d_eff"],
             spans_tp=tp["spans"],
             spanning_frac_tp=tp["spanning_pore_frac"],
             src_frac_tp=tp["source_connected_frac"],
             conv_tp=tp["converged"], iter_tp=tp["n_iter"])
    m.update(tau_ip=ip["tau"], d_eff_ip=ip["d_eff"],
             spans_ip=ip["spans"],
             spanning_frac_ip=ip["spanning_pore_frac"],
             conv_ip=ip["converged"], iter_ip=ip["n_iter"])
    if np.isfinite(tp["tau"]) and np.isfinite(ip["tau"]) \
            and ip["tau"] > 0:
        m["tau_anisotropy"] = tp["tau"] / ip["tau"]
    else:
        m["tau_anisotropy"] = np.nan
    m.update({f"si_{k}": v for k, v in
              accessible_fraction(vol).items()})
    m["voxel_nm"] = voxel_nm
    m["volume_um3"] = float(vol.size * voxel_nm ** 3 / 1e9)
    return m
