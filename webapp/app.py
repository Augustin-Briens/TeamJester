"""Batch-classifier web app for the Hackathon-Polaron SEM batches.

Wraps the analysis pipeline (segment -> features -> robust Mahalanobis /
nearest-centroid model trained on the 31 labelled images). Upload one or
more SEM photos (img_*_BSE.tif style, or any image); the API segments each
one with the identical pipeline settings, computes the explainable features,
and returns:

- IN/OUT of the Batch_3 baseline envelope (distance vs threshold),
- most-like batch (1/2/3) with confidence %,
- top-3 drivers in plain words,
- the segmentation overlay as base64 PNG.

Group calls (median of uploaded images) use the group-size-adaptive
threshold, mirroring categorise.py's classify_folder.
"""
import base64
import io
import os
import sys
import uuid

import numpy as np
import pandas as pd
import tifffile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import common as C            # noqa: E402
import segment as S           # noqa: E402
import features as FT         # noqa: E402
import categorise as CAT      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THUMB_DIR = os.path.join(HERE, "outputs", "thumbs")
UP_DIR = os.path.join(C.OUT, "uploads")
os.makedirs(UP_DIR, exist_ok=True)

_EMB = None  # lazy: torch+dinov2 import is heavy


def _emb():
    global _EMB
    if _EMB is None:
        try:
            import embclassify as E
            E.embed_path  # attr check
            _EMB = E
        except Exception:
            _EMB = False
    return _EMB

ASSUME = ("The electrode is believed to be graphite (grey in BSE) with "
          "brighter silicon-based particles; black is pore or crack. This "
          "comes from image appearance and is unconfirmed.")

# --------------------------------------------------------------------------
# model bootstrap (once at startup)
# --------------------------------------------------------------------------
F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
FULL = F_[F_.subset == "full"].reset_index(drop=True)
REP = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
STATS = pd.read_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv"))
KEPT = STATS.feature.tolist()
SAFE = [f for f in KEPT if f not in CAT.ARTEFACT_RISK]
FEATS = CAT.select_features(FULL, SAFE, REP)
MODEL = CAT.build_model(FULL, FEATS)
B3 = FULL[FULL.batch == C.BASELINE]
_thr1, _ = CAT.per_image_threshold(B3, MODEL) \
    if hasattr(CAT, "per_image_threshold") else (None, None)
if _thr1 is None:
    _ds = [CAT.dist_and_predict(r, MODEL)[0] for _, r in B3.iterrows()]
    THR1 = float(np.nanpercentile(_ds, CAT.THRESH_Q * 100))
else:
    THR1 = float(_thr1)

SEM_SHAPE = (2148, 7000)      # dataset image shape (h, w) for sanity checks
PX_ASSUMED_UM = 0.025         # fallback pixel size when metadata missing

# Focused Batch_2-vs-Batch_3 discrimination (validated 87.5% LOO on the
# all-feature pool; ~67% on artefact-safe features only — the gap is how
# much of the separation rides on imaging signatures).
from sklearn.linear_model import LogisticRegression  # noqa: E402

_NEW_FEATS_CSV = os.path.join(C.TABLE_DIR, "exp_newfeat.csv")
_B23 = {}
if os.path.exists(_NEW_FEATS_CSV):
    _E = pd.read_csv(_NEW_FEATS_CSV)
    _E = _E[_E.subset == "full"][["image_id", "elong_pore_n_mm2",
                                  "si_border_pore", "bse_grad_coh",
                                  "inl_lbp_ent", "inl_lbp_flat",
                                  "inl_grad_coh", "etd_fft_hi"]]
    _FB = FULL.merge(_E, on="image_id")
    _B23 = _FB[_FB.batch.isin(["Batch_2", "Batch_3"])].reset_index(drop=True)
    _B23Y = (_B23.batch == "Batch_2").astype(int).values

    def _fit_b23(feats):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(_B23[feats].values.astype(float), _B23Y)
        return clf, feats

    B23_ALL = _fit_b23(["bse_bulk_texture", "inl_lbp_flat",
                        "elong_pore_n_mm2", "etd_roughness",
                        "gr_st_coherence"])
    B23_SAFE = _fit_b23(["gr_st_coherence", "inlens_edge_density",
                         "elong_pore_n_mm2", "inl_lbp_flat",
                         "pore_thick_d10_um"])


def _detect_role(name):
    n = os.path.basename(name).lower()
    if "bse" in n:
        return "bse"
    if "inlens" in n:
        return "inlens"
    if "_etd" in n or "_se" in n or "etd" in n or "_se." in n:
        return "etd"
    return "bse"


def _to_tiff(raw, dst):
    """Normalise an upload to a TIFF carrying inch-resolution tags so
    pixel_size_um() works. Returns (path, pixel_source)."""
    lo = os.path.splitext(dst)[1].lower()
    if lo in (".tif", ".tiff"):
        with open(dst, "wb") as f:
            f.write(raw)
        try:
            C.pixel_size_um(dst)
            return dst, "metadata"
        except Exception:
            pass
    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.asarray(img)
    # 25.4e3 um/inch / 0.025 um/px -> 1,016,000 dpi
    tifffile.imwrite(dst, arr, resolution=(1016000, 1016000),
                     resolutionunit="INCH")
    return dst, "assumed 25 nm/px (no usable metadata)"


def _analyse_one(path_bse, path_in, path_etd, iid):
    rec = dict(batch="unseen", image_id=iid, bse=path_bse,
               inlens=path_in, topo=path_etd,
               topo_kind="ETD" if path_etd else None)
    g, crop_note = C.load_gray(rec["bse"])
    il = S.load_detector(rec)
    r = S.segment(g, il)
    np.savez_compressed(os.path.join(
        C.MASK_DIR, f"unseen_{iid}_masks.npz"),
        pore=r["pore"], silicon=r["silicon"],
        labels=r["labels"].astype(np.int32),
        pore_L=r["pore"], silicon_L=r["silicon"],
        labels_L=r["labels"].astype(np.int32),
        pore_R=r["pore"], silicon_R=r["silicon"],
        labels_R=r["labels"].astype(np.int32))
    fe, _, _ = FT.compute(rec, "full")
    d, best, conf, drivers, dists = CAT.dist_and_predict(
        pd.Series(fe), MODEL)
    call = "IN" if d <= THR1 else "OUT"
    out = S.overlay_image(rec)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    ov64 = base64.b64encode(buf.getvalue()).decode()
    warn = []
    if g.shape != SEM_SHAPE:
        warn.append(f"size {g.shape[1]}x{g.shape[0]} differs from the "
                    f"dataset's {SEM_SHAPE[1]}x{SEM_SHAPE[0]}")
    b23 = None
    if _B23 is not None and hasattr(_B23, "__len__") and len(_B23):
        b23 = {}
        for tag, (clf, feats) in (("all", B23_ALL), ("safe", B23_SAFE)):
            avail = [f for f in feats if f in fe and np.isfinite(fe[f])]
            if len(avail) == len(feats):
                x = np.array([fe[f] for f in feats],
                             dtype=float).reshape(1, -1)
                p = clf.predict_proba(x)[0]
                b23[tag] = dict(p_b2=float(p[1]), p_b3=float(p[0]),
                                call="Batch_2" if p[1] > p[0] else "Batch_3",
                                feats=feats)
    return dict(
        image_id=iid, shape=list(g.shape), px_source=None,
        dist=float(d), threshold=THR1, call_inout=call,
        most_like=best, confidence=float(conf),
        drivers=[dict(feature=f, plain=CAT.PLAIN_WORDS.get(f, f),
                      z=float(z)) for f, z in drivers],
        centroid_distances={k: float(v) for k, v in dists.items()},
        features={k: float(fe[k]) for k in list(FEATS) + 
                  [f for f in (B23_ALL[1] + B23_SAFE[1])
                   if f not in FEATS] if k in fe and np.isfinite(fe[k])},
        b23=b23, overlay_png=ov64, warnings=warn)


app = FastAPI(title="SEM batch classifier")


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if not files:
        return JSONResponse({"error": "no files"}, status_code=400)
    job = os.path.join(UP_DIR, uuid.uuid4().hex[:12])
    os.makedirs(job)
    warnings = []
    # deterministic grouping by name stem: *_Inlens/_ETD/_SE files attach to
    # the matching *_BSE location; unmarked images are treated as BSE.
    byname = {}
    for uf in files:
        raw = None  # already read; re-read via uf.file
        uf.file.seek(0)
        raw = uf.file.read()
        role = _detect_role(uf.filename)
        stem = os.path.splitext(os.path.basename(uf.filename))[0]
        key = stem.split("_Inlens")[0].split("_ETD")[0].split("_SE")[0] \
                .split("_BSE")[0]
        byname.setdefault(key, {})[role] = raw
    items = []
    used_ids = set()
    for key, roles in byname.items():
        if "bse" not in roles:
            continue
        iid = uuid.uuid4().hex[:8]
        bpath = os.path.join(job, f"img_{iid}_BSE.tif")
        bpath, pxs = _to_tiff(roles["bse"], bpath)
        ipath = epath = None
        if "inlens" in roles:
            ipath = os.path.join(job, f"img_{iid}_Inlens.tif")
            ipath, _ = _to_tiff(roles["inlens"], ipath)
        if "etd" in roles:
            epath = os.path.join(job, f"img_{iid}_ETD.tif")
            epath, _ = _to_tiff(roles["etd"], epath)
        it = _analyse_one(bpath, ipath, epath, iid)
        it["px_source"] = pxs
        it["detectors"] = [r for r in ("bse", "inlens", "etd") if r in roles]
        it["_epath"] = epath
        items.append(it)
        used_ids.add(iid)
    if not items:
        return JSONResponse(
            {"error": "no BSE image found; upload at least one image "
                      "(name it *_BSE.tif to mark it as the BSE detector, "
                      "or upload any single SEM photo)"}, status_code=400)
    # group-level call
    fdf = pd.DataFrame([
        {k: v for k, v in it["features"].items()} for it in items])
    med = fdf.apply(pd.to_numeric, errors="coerce").median()
    thrn = CAT.group_threshold(B3, MODEL, len(items))
    d, best, conf, drivers, dists = CAT.dist_and_predict(med, MODEL)
    group = dict(
        n_images=len(items), dist=float(d), threshold=float(thrn),
        call_inout="IN" if d <= thrn else "OUT",
        most_like=best, confidence=float(conf),
        drivers=[dict(feature=f, plain=CAT.PLAIN_WORDS.get(f, f),
                      z=float(z)) for f, z in drivers],
        centroid_distances={k: float(v) for k, v in dists.items()},
        model_features=FEATS)
    # DINOv2 topography-embedding second opinion ----------------------------
    emb_block = None
    E = _emb()
    epaths = [it["_epath"] for it in items if it.get("_epath")]
    if E and epaths:
        try:
            embs = np.stack([E.embed_path(p) for p in epaths])
            r = E.classify_emb(embs)
            thr = E.group_thr(len(epaths))
            neigh = []
            for nb in r["neighbors"]:
                tp = os.path.join(THUMB_DIR, f"{nb['iid']}_etd.png")
                b64 = None
                if os.path.exists(tp):
                    with open(tp, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                neigh.append(dict(image_id=nb["iid"], batch=nb["batch"],
                                  thumb_png=b64))
            emb_block = dict(
                n_images=len(epaths), dist=float(r["dist"]),
                threshold=float(thr),
                call_inout="IN" if r["dist"] <= thr else "OUT",
                most_like=r["most_like"], confidence=float(r["conf"]),
                centroid_distances={k: float(v)
                                    for k, v in r["dists"].items()},
                neighbors=neigh,
                model="dinov2-small ETD embeddings -> PCA -> "
                      "robust-Mahalanobis + nearest centroid",
                note="Embedding call: not human-word explainable; the "
                     "nearest dataset patches below are the evidence.")
        except Exception as e:
            warnings.append(f"embedding classifier failed: {e}")
    elif not epaths:
        warnings.append("no ETD/SE topography image: embedding second "
                        "opinion unavailable (it needs the topography "
                        "detector).")
    if len(items) < 5:
        warnings.append(
            f"{len(items)} image(s) uploaded: single-image calls are "
            f"unreliable (~45% leave-one-out accuracy). Upload 5+ images "
            f"of the same sample for the reliable group call.")
    for it in items:
        it.pop("_epath", None)
    if not any(r == "inlens" for it in items for r in it["detectors"]):
        warnings.append("no Inlens detector: fine-crack features missing.")
    if not any(r == "etd" for it in items for r in it["detectors"]):
        warnings.append("no ETD detector: topography features missing.")
    return dict(assumption=ASSUME, group=group, images=items,
                warnings=warnings, embedding=emb_block,
                features_used=FEATS,
                loo_note="Validation: single-image LOO ~45% (shuffled "
                         "control ~37%); group-of-5+ reaches ~97-100% for "
                         "Batch_1-like calls. Focused B2-vs-B3 logreg "
                         "(per-image, all-features pool): 87.5% LOO "
                         "accuracy — but it leans on the two "
                         "noise-correlated imaging features; the "
                         "artefact-safe pool reaches ~67%.")


@app.get("/api/health")
def health():
    return dict(ok=True, n_features=len(FEATS), threshold_1img=THR1)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/{path:path}")
def static_files(path: str):
    p = os.path.join(HERE, "static", path)
    if os.path.isfile(p):
        return FileResponse(p)
    return FileResponse(os.path.join(HERE, "static", "index.html"))
