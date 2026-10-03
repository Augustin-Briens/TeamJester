"""DINOv2-embedding batch classifier (second opinion to categorise.py).

Pipeline (same machinery as categorise.py, features replaced by embedding
PCA scores): dinov2-small CLS+patch-mean over 3 square crops of the ETD/SE
topography detector -> robust-z vs Batch_3 -> PCA(6) -> robust-Mahalanobis
distance + nearest centroid.

Explainability: embedding dims are not human-named, so the call is reported
together with the K nearest dataset images in embedding space (instance-
based explanation). The ETD-detector variant was validated: group-of-5
accuracy 100% (B1) / 100% (B2) vs 97%/70% for classical features; single
images still unreliable (55% LOO). corr(emb distance, noise_mad)=0.17 ->
not an imaging-noise proxy.

Model artefacts (webapp/emb_model.npz): mu, sc, P (PCA), cents, per-image
PCA scores + ids + batches (for neighbour lookup), group thresholds n=1..8.

Usage:
    python3 embclassify.py build        # fit + save model + thumbs
    python3 embclassify.py <etd.tif>    # classify one topography image
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import categorise as CAT

MODEL_ID = "facebook/dinov2-small"
N_PCA = 6
N_CROPS = 3
K_NEIGH = 3
WEBAPP = os.path.normpath(os.path.join(HERE, "..", "webapp"))
MODEL_NPZ = os.environ.get(
    "EMB_MODEL_NPZ",
    os.path.join(HERE, "outputs", "emb_model.npz"))
THUMB_DIR = os.path.join(WEBAPP, "outputs", "thumbs")

_model = None
_net = None


# --------------------------------------------------------------------------
def net():
    global _net
    if _net is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        mod = AutoModel.from_pretrained(MODEL_ID).eval()
        _net = (proc, mod, torch)
    return _net


def embed_gray(g):
    """Mean CLS+patchmean embedding over N_CROPS square crops."""
    proc, mod, torch = net()
    from PIL import Image
    h, w = g.shape
    s = min(h, w)
    xs = np.linspace(0, max(w - s, 0), N_CROPS).astype(int) if w > h else [0]
    outs = []
    for x in xs:
        img = Image.fromarray(g[:, x:x + s].astype(np.uint8)).convert("RGB")
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            o = mod(**inp)
        outs.append(np.concatenate(
            [o.last_hidden_state[0, 0].numpy(),
             o.last_hidden_state[0, 1:].mean(0).numpy()]))
    return np.mean(outs, 0)


def embed_path(path):
    g, _ = C.load_gray(path)
    return embed_gray(g)


# --------------------------------------------------------------------------
def build_model_artifacts():
    """Embed every dataset topography image, fit z->PCA->centroids on the
    whole dataset (same protocol as categorise's images_needed), save."""
    locs = C.find_locations()
    rows = []
    for rec in locs:
        if not rec.get("topo"):
            continue
        e = embed_path(rec["topo"])
        rows.append(dict(iid=rec["image_id"], batch=rec["batch"], emb=e))
        C.log(f"emb {rec['batch']} {rec['image_id']}")
    ids = [r["iid"] for r in rows]
    batches = np.array([r["batch"] for r in rows])
    X = np.stack([r["emb"] for r in rows])
    b3 = X[batches == C.BASELINE]
    mu = np.median(b3, 0)
    sc = np.percentile(b3, 75, 0) - np.percentile(b3, 25, 0) + 1e-9
    Z = (X - mu) / sc
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Vt[:N_PCA].T
    T = Zc @ P
    df = pd.DataFrame(T, columns=[f"pc{j}" for j in range(N_PCA)])
    df["batch"] = batches
    df["image_id"] = ids
    feats = df.columns[:N_PCA].tolist()
    model = CAT.build_model(df, feats)
    b3df = df[df.batch == C.BASELINE]
    thrs = {n: CAT.group_threshold(b3df, model, n) for n in range(1, 9)}
    np.savez(MODEL_NPZ,
             ids=np.array(ids), batches=batches, T=T, mu=mu, sc=sc, P=P,
             zmean=Z.mean(0),
             mu2=model["mu"][feats].values.astype(float),
             scale2=model["scale"][feats].values.astype(float),
             cov=model["cov"],
             cents=np.stack([model["cents"][b] for b in C.BATCHES]),
             cent_batches=np.array(C.BATCHES),
             thrs=np.array([thrs[n] for n in range(1, 9)]))
    make_thumbs(locs)
    C.log(f"saved {MODEL_NPZ} + thumbs")


def make_thumbs(locs):
    from PIL import Image
    os.makedirs(THUMB_DIR, exist_ok=True)
    for rec in locs:
        if not rec.get("topo"):
            continue
        out = os.path.join(THUMB_DIR, f"{rec['image_id']}_etd.png")
        if os.path.exists(out):
            continue
        g, _ = C.load_gray(rec["topo"])
        h, w = g.shape
        crop = g[:, (w - h) // 2:(w - h) // 2 + h]
        Image.fromarray(crop.astype(np.uint8)).resize(
            (256, 256)).save(out)


def load_model():
    global _model
    if _model is None:
        z = np.load(MODEL_NPZ, allow_pickle=True)
        cents = {b: c for b, c in zip(z["cent_batches"].tolist(),
                                      z["cents"])}
        _model = dict(z=z, cents=cents,
                      feats=[f"pc{j}" for j in range(z["P"].shape[1])])
    return _model


def classify_emb(emb):
    """One topography image (or group, as (n,768) matrix -> median pool)
    -> dict(dist, call, most_like, conf, neighbors)."""
    m = load_model()
    z = m["z"]
    if emb.ndim == 2:
        emb = np.median(emb, 0)
    zz = (emb - z["mu"]) / z["sc"]
    t = (zz - z["zmean"]) @ z["P"]
    T_train = z["T"]
    # identical machinery to categorise.py: robust-z -> Mahalanobis to B3
    feats = m["feats"]
    model = dict(feats=feats,
                 mu=pd.Series(z["mu2"], index=feats),
                 scale=pd.Series(z["scale2"], index=feats),
                 cov=z["cov"],
                 cents={b: c for b, c in
                        zip(z["cent_batches"].tolist(), z["cents"])})
    row = pd.Series(t, index=feats)
    d, best, conf, drivers, dists = CAT.dist_and_predict(row, model)
    dnb = np.linalg.norm(T_train - t, axis=1)
    nn = np.argsort(dnb)[:K_NEIGH]
    neigh = [dict(iid=str(z["ids"][i]), batch=str(z["batches"][i]),
                  dist=float(dnb[i])) for i in nn]
    return dict(dist=d, dists=dists, most_like=best, conf=float(conf),
                drivers=drivers, neighbors=neigh, t=t)


def group_thr(n):
    z = load_model()["z"]
    return float(z["thrs"][min(max(n, 1), 8) - 1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_model_artifacts()
    elif len(sys.argv) > 1:
        e = embed_path(sys.argv[1])
        r = classify_emb(e)
        print(r["most_like"], round(r["conf"], 3), "dist",
              round(r["dist"], 2), "thr", round(group_thr(1), 2),
              "->", "IN" if r["dist"] <= group_thr(1) else "OUT",
              "neighbors:", [(n["iid"], n["batch"]) for n in r["neighbors"]])
