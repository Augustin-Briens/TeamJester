"""Modal runner for recon3d — SliceGAN training in parallel containers.

    .venv/bin/python -m modal run modal_recon3d.py              # full (CPU)
    .venv/bin/python -m modal run modal_recon3d.py --fast       # dev
    RECON3D_GPU=1 .venv/bin/python -m modal run modal_recon3d.py --gpu
        (GPU needs a payment method on file — credits alone don't
         unlock GPU functions; CPU runs on credits)
    .venv/bin/python -m modal run modal_recon3d.py --iters 8000 --seeds 0,1,2

Masks are built locally under the baseline-fitted recipe (same ruler
as micro2dfn's recipe.json) and shipped to the workers as npz bytes —
the remote image only needs torch + numpy.  Each (batch, seed) trains
in its own container; every finished model writes its volumes + loss
history into the Modal Volume `polaron-recon3d`, so a client
disconnect or preemption never loses completed work — rerunning the
command resumes where it left off (finished tags are skipped) and then
the analysis stage runs locally into recon3d_output/.
"""
import os

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.8.0", "numpy==2.5.3")
    .add_local_python_source("recon3d")
)

app = modal.App("polaron-recon3d", image=image)

# Durable remote store: remote functions persist each finished model's
# volumes + history here; the local entrypoint downloads whatever is
# missing, and reruns skip tags that already exist remotely.
R3VOL = modal.Volume.from_name("polaron-recon3d", create_if_missing=True)
MNT = "/r3"


def _train_impl(masks_blob: bytes, batch: str, seed: int, iters: int,
                V: int, n_vols: int, tag: str) -> dict:
    """Train one SliceGAN on one batch's masks; persist volumes + hist
    to the mounted volume (crash-resumable) and return the history."""
    import io as _io
    import json

    import numpy as np
    import torch
    torch.set_num_threads(8)
    from recon3d import data, slicegan

    masks = data.unpack_masks(masks_blob)[batch]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = slicegan.train(masks, seed=seed, iters=iters, V=V,
                           device=device)
    vols = slicegan.generate(state, n_vols, seed=1000 + seed,
                             device=device)

    os.makedirs(f"{MNT}/volumes", exist_ok=True)
    os.makedirs(f"{MNT}/models", exist_ok=True)
    with open(f"{MNT}/volumes/{tag}.npz", "wb") as fh:
        np.savez_compressed(
            fh, volumes=vols,
            meta=np.array([dict(batch=batch, seed=seed, V=V,
                                iters=iters, device=device,
                                torch_version=torch.__version__)],
                          dtype=object))
    with open(f"{MNT}/models/{tag}.json", "w") as fh:
        json.dump(state["hist"], fh)
    R3VOL.commit()
    return {"hist": state["hist"]}


# CPU is the default path — runs on any workspace incl. free credits.
# GPU requires a payment method on file: Modal validates EVERY GPU
# function in the app at deploy time (even unused ones), so the GPU
# function only exists when RECON3D_GPU=1 is in the environment.
@app.function(memory=16384, timeout=3 * 3600, max_containers=16,
              retries=2, volumes={MNT: R3VOL})
def train_cpu(masks_blob: bytes, batch: str, seed: int, iters: int,
              V: int, n_vols: int, tag: str) -> dict:
    return _train_impl(masks_blob, batch, seed, iters, V, n_vols, tag)


if os.environ.get("RECON3D_GPU"):
    train_gpu = app.function(
        gpu="A10G", memory=16384, timeout=3 * 3600, max_containers=16,
        retries=2, volumes={MNT: R3VOL})(_train_impl)


def _remote_files(subdir: str) -> set[str]:
    """Basenames already persisted in the volume."""
    try:
        return {os.path.basename(e.path) for e in
                R3VOL.listdir(subdir)}
    except Exception:
        return set()


def _read_all(path: str) -> bytes:
    """Sync-safe volume read — read_file returns an async generator in
    modal>=1.6."""
    stream = R3VOL.read_file(path)
    if hasattr(stream, "__aiter__"):
        import asyncio

        async def _go() -> bytes:
            buf = b""
            async for ch in stream:
                buf += ch
            return buf
        return asyncio.run(_go())
    return b"".join(stream)


def _pull(tag: str, out: str) -> None:
    """Download volumes/{tag}.npz + models/{tag}.json if missing
    locally."""
    npz = os.path.join(out, "volumes", f"{tag}.npz")
    if not os.path.exists(npz):
        with open(npz, "wb") as fh:
            fh.write(_read_all(f"volumes/{tag}.npz"))
    import json
    import pandas as pd
    csv = os.path.join(out, "models", f"{tag}.csv")
    if not os.path.exists(csv):
        hist = json.loads(_read_all(f"models/{tag}.json"))
        pd.DataFrame(hist).assign(
            seconds=hist.get("train_seconds")).to_csv(csv,
                                                     index=False)


@app.local_entrypoint()
def main(out: str = "recon3d_output", fast: bool = False,
         iters: int = 0, volume: int = 0, seeds: str = "",
         vols: int = 0, batches: str = "", downsample: int = 0,
         baseline: str = "Batch_3", gpu: bool = False):
    import json
    import time

    import run_recon3d as R
    from recon3d import analysis, config, data

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    batch_list = R.get_batches(ROOT, batches or None)
    V = volume or (64 if fast else config.VOLUME)
    it = iters or (config.ITERS_FAST if fast else config.ITERS)
    sd = ([int(s) for s in seeds.split(",")] if seeds else
          list(config.SEEDS_FAST if fast else config.SEEDS))
    nv = vols or (config.VOLUMES_FAST if fast
                  else config.VOLUMES_PER_SEED)
    ds = downsample or config.DOWNSAMPLE

    masks, prov = R.build_or_load_masks(ROOT, out, batch_list, ds,
                                        baseline)
    tag = prov["tag"]
    settings = dict(V=V, voxel_nm=25.0 * ds, iters=it,
                    seeds=sd, vols_per_seed=nv, n_seeds=len(sd),
                    n_volumes=len(sd) * nv,
                    mode="modal starmap " + ("A10G" if gpu else "cpu"),
                    fast=bool(fast), downsample=ds,
                    baseline=baseline, ruler=tag,
                    ruler_info={k: v for k, v in prov.items()
                                if k != "tag"})
    with open(os.path.join(out, "run_settings.json"), "w") as fh:
        json.dump(settings, fh, indent=1)

    if gpu and not os.environ.get("RECON3D_GPU"):
        raise SystemExit(
            "--gpu needs RECON3D_GPU=1 in the environment: Modal "
            "validates GPU functions at deploy time, so the A10G "
            "function is only added to the app when the env var is "
            "set (a payment method on file is still required).")

    # expected artifacts; skip tags already persisted remotely or
    # downloaded locally — makes every rerun a resume
    expected = [(b, s, f"{b}__s{s}__V{V}i{it}__r{tag}")
                for b in batch_list for s in sd]
    remote_npz = _remote_files("volumes")
    jobs = []
    for b, s, mtag in expected:
        local = os.path.exists(os.path.join(out, "volumes",
                                            f"{mtag}.npz"))
        remote = f"{mtag}.npz" in remote_npz
        if local or remote:
            print(f"  {mtag}: {'cached locally' if local else 'in volume'}, "
                  "skipping")
            continue
        jobs.append((data.pack_masks({b: masks[b]}), b, s, it, V,
                     nv, mtag))

    if jobs:
        print(f"{len(jobs)} training jobs on Modal "
              f"({'A10G' if gpu else 'cpu'}, parallel)")
        fn = train_gpu if gpu else train_cpu
        t0 = time.time()
        try:
            for res in fn.starmap(jobs):
                pass          # results are persisted remotely
        except Exception as exc:
            print(f"starmap interrupted: {exc}\nremote artifacts are "
                  "preserved in volume 'polaron-recon3d' — rerun this "
                  "command to resume")
        print(f"training wall time {time.time() - t0:.0f}s")

    # pull whatever is persisted but not yet local
    remote_npz = _remote_files("volumes")
    missing = []
    for b, s, mtag in expected:
        if os.path.exists(os.path.join(out, "volumes", f"{mtag}.npz")):
            continue
        if f"{mtag}.npz" in remote_npz:
            _pull(mtag, out)
            print(f"  {mtag}: pulled from volume")
        else:
            missing.append(mtag)
    if missing:
        print(f"still missing {len(missing)}: {missing} — rerun to "
              "resume training")
        return

    analysis.run_analysis(out, batch_list, masks, settings)
