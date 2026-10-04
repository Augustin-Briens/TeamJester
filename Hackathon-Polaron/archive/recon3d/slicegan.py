"""SliceGAN: 2-D -> 3-D microstructure reconstruction (Kench & Cooper 2021).

Generator produces a 3-D volume; the critic only ever sees 2-D slices of
it.  The generator therefore learns to make a volume whose slices are
indistinguishable from the real segmented cross-sections.

Single-view anisotropy handling: all real images share one orientation
(row = through-plane z, col = in-plane x).  The critic is shown ONLY
slices that contain the z axis — generated (z,x) and (z,y) planes —
never top-down (y,x) planes.  This fixes the through-plane statistics
from data while treating the two in-plane directions as statistically
equivalent (in-plane isotropy assumption — documented in report.md).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from . import config


def _layers(V: int) -> int:
    """ConvTranspose2x layers needed to go 4 -> V.  V must be a power of
    two >= 8."""
    if V < 8 or (V & (V - 1)):
        raise ValueError(f"VOLUME must be a power of two >= 8, got {V}")
    return int(math.log2(V)) - 2


def build_nets(V: int = config.VOLUME, latent: int = config.LATENT,
               g_ch: int = config.G_CH, d_ch: int = config.D_CH,
               n_ph: int = config.N_PHASES):
    import torch
    import torch.nn as nn

    L = _layers(V)

    g_blocks = []
    cin = latent
    for i in range(L - 1):
        g_blocks += [
            nn.ConvTranspose3d(cin, g_ch, 4, 2, 1, bias=False),
            nn.BatchNorm3d(g_ch),
            nn.ReLU(True),
        ]
        cin = g_ch
    g_blocks += [nn.ConvTranspose3d(cin, n_ph, 4, 2, 1, bias=True)]
    G = nn.Sequential(*g_blocks)

    d_blocks = []
    cin = n_ph
    for i in range(L):
        cout = d_ch if i < L - 1 else d_ch
        d_blocks += [
            nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        cin = cout
    d_blocks += [nn.Conv2d(cin, 1, 4, 1, 0, bias=True)]   # 4x4 -> score
    D = nn.Sequential(*d_blocks)
    return G, D


def _one_hot(m: "torch.Tensor", n_ph: int) -> "torch.Tensor":
    import torch
    return torch.nn.functional.one_hot(
        m.long(), n_ph).permute(0, 3, 1, 2).float()


class RealSampler:
    """Uniform random V x V crops from a batch's masks (numpy, CPU)."""

    def __init__(self, masks: dict[str, np.ndarray], V: int, seed: int):
        self.imgs = [m for m in masks.values()
                     if m.shape[0] >= V and m.shape[1] >= V]
        if not self.imgs:
            raise ValueError("no mask large enough for V x V crops")
        self.V, self.rng = V, np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        V, rng = self.V, self.rng
        picks = rng.integers(0, len(self.imgs), n)
        out = np.empty((n, V, V), np.uint8)
        for k, i in enumerate(picks):
            m = self.imgs[i]
            y = rng.integers(0, m.shape[0] - V + 1)
            x = rng.integers(0, m.shape[1] - V + 1)
            c = m[y:y + V, x:x + V]
            # in-plane flip augmentation only — never flip z
            if rng.random() < 0.5:
                c = c[:, ::-1]
            out[k] = c
        return out


def _fake_slices(vol, axes: tuple, rng) -> "torch.Tensor":
    """One random slice per volume per axis in `axes`.

    vol: (B, C, Z, Y, X) softmax probabilities.
    axis 1 -> (Z,X) planes; axis 2 -> (Z,Y) planes.  Both contain the
    through-plane axis; axis-0 (Y,X top-down) is never judged.
    """
    import torch
    V = vol.shape[-1]
    out = []
    for a in axes:
        idx = int(rng.integers(0, V))
        out.append(vol.select(dim=2 + a, index=idx))
    return torch.cat(out, dim=0)


def _gp(D, real, fake, device):
    import torch
    eps = torch.rand(real.shape[0], 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = D(interp).sum()
    grad = torch.autograd.grad(score, interp, create_graph=True)[0]
    return ((grad.reshape(grad.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()


def train(masks: dict[str, np.ndarray], seed: int = 0,
          iters: int = config.ITERS, V: int = config.VOLUME,
          device: str | None = None, verbose: bool = True) -> dict:
    """WGAN-GP training on one batch's masks.  Returns the trained nets
    and a light loss history (JSON-safe)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 77)
    G, D = build_nets(V)
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), config.LR,
                             (config.BETA1, config.BETA2))
    opt_d = torch.optim.Adam(D.parameters(), config.LR,
                             (config.BETA1, config.BETA2))
    real_ds = RealSampler(masks, V, seed)
    hist = {"iter": [], "d": [], "g": [], "w": []}
    axes = config.SLICE_AXES
    t0 = time.time()
    for it in range(1, iters + 1):
        for _ in range(config.N_CRITIC):
            z = torch.randn(config.BATCH, config.LATENT, 4, 4, 4,
                            device=device)
            with torch.no_grad():
                vol = torch.softmax(G(z), dim=1)
            fake = _fake_slices(vol, axes, rng).detach()
            real = _one_hot(
                torch.from_numpy(
                    real_ds.sample(fake.shape[0])).to(device),
                config.N_PHASES)
            d_real = D(real).mean()
            d_fake = D(fake).mean()
            loss_d = d_fake - d_real + config.LAMBDA_GP * _gp(
                D, real, fake, device)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
        z = torch.randn(config.BATCH, config.LATENT, 4, 4, 4,
                        device=device)
        vol = torch.softmax(G(z), dim=1)
        fake = _fake_slices(vol, axes, rng)
        loss_g = -D(fake).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if it % config.PRINT_EVERY == 0 or it == 1:
            hist["iter"].append(it)
            hist["d"].append(float(loss_d.item()))
            hist["g"].append(float(loss_g.item()))
            hist["w"].append(float((d_real - d_fake).item()))
            if verbose:
                s = (it / (time.time() - t0))
                print(f"    it {it}/{iters}  D {loss_d.item():+.3f}  "
                      f"W {(d_real - d_fake).item():+.3f}  "
                      f"({s:.1f} it/s)", flush=True)
    hist["device"] = device
    hist["train_seconds"] = round(time.time() - t0, 1)
    return {"G": G.state_dict(), "D": D.state_dict(), "hist": hist,
            "V": V, "seed": seed}


def generate(state: dict, n_volumes: int, seed: int = 0,
             device: str | None = None) -> np.ndarray:
    """Sample n_volumes argmax label volumes (n, Z, Y, X) uint8."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    V = state["V"]
    G, _ = build_nets(V)
    G.load_state_dict(state["G"])
    G = G.to(device).eval()
    torch.manual_seed(seed)
    vols = []
    with torch.no_grad():
        for _ in range(n_volumes):
            z = torch.randn(1, config.LATENT, 4, 4, 4, device=device)
            v = torch.softmax(G(z), dim=1).argmax(1)[0]
            vols.append(v.cpu().numpy().astype(np.uint8))
    return np.stack(vols)


def save_volumes(vols: np.ndarray, path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, volumes=vols,
                        meta=np.array([meta], dtype=object))


def load_volumes(path: str) -> tuple[np.ndarray, dict]:
    z = np.load(path, allow_pickle=True)
    return z["volumes"], dict(z["meta"][0])
