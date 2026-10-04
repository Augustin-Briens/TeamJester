"""Experiment: DINOv2 embeddings for batch classification.

Per image: 3 square crops (left/center/right) per available detector,
dinov2-small CLS + patch-mean embeddings, aggregated per detector.
Classifier: robust-z vs Batch_3 -> PCA (fit inside each LOO fold) -> same
robust-Mahalanobis + nearest-centroid machinery as categorise.py.

Outputs: outputs/tables/exp_dino_emb.npz, exp_dino_results.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import categorise as CAT

MODEL_ID = "facebook/dinov2-small"
N_PCA = 6
EMB_OUT = os.path.join(C.TABLE_DIR, "exp_dino_emb.npz")
RES_OUT = os.path.join(C.TABLE_DIR, "exp_dino_results.csv")


def crops_square(g, n=3):
    h, w = g.shape
    s = min(h, w)
    if w <= h:
        return [g[:s, :s]]
    xs = np.linspace(0, w - s, n).astype(int)
    return [g[:, x:x + s] for x in xs]


def embed_all(locs):
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    keys, embs = [], []
    for rec in locs:
        for det in ("bse", "inlens", "topo"):
            path = rec.get(det)
            if not path:
                continue
            g, _ = C.load_gray(path)
            cs = crops_square(g)
            from PIL import Image
            outs = []
            for c in cs:
                img = Image.fromarray(c.astype(np.uint8)).convert("RGB")
                inp = proc(images=img, return_tensors="pt")
                with torch.no_grad():
                    o = model(**inp)
                cls = o.last_hidden_state[0, 0].numpy()
                pm = o.last_hidden_state[0, 1:].mean(0).numpy()
                outs.append(np.concatenate([cls, pm]))
            e = np.mean(outs, 0)
            keys.append(f"{rec['image_id']}|{det}")
            embs.append(e)
        C.log(f"emb {rec['batch']} {rec['image_id']}")
    return keys, np.array(embs)


def load_or_compute(locs):
    if os.path.exists(EMB_OUT):
        z = np.load(EMB_OUT)
        return z["keys"].tolist(), z["embs"]
    keys, embs = embed_all(locs)
    np.savez(EMB_OUT, keys=np.array(keys), embs=embs)
    return keys, embs


def run_loo(key_batch, keys, embs, dets=("bse", "inlens", "topo"), n_pca=N_PCA):
    """LOO over images using embeddings of the given detectors."""
    import re
    idx = [i for i, k in enumerate(keys)
           if k.split("|")[1] in dets]
    # per-image mean over chosen detectors
    ids = sorted({keys[i].split("|")[0] for i in idx})
    X = np.stack([np.mean([embs[i] for i in idx
                           if keys[i].split("|")[0] == iid], 0)
                  for iid in ids])
    y = np.array([key_batch[i] for i in ids])
    preds, d_out, thr_l = [], [], []
    for i in range(len(ids)):
        tr = np.arange(len(ids)) != i
        Xtr, Xte = X[tr], X[i:i + 1]
        ytr = y[tr]
        # robust z within fold, on B3
        b3 = Xtr[ytr == C.BASELINE]
        mu = np.median(b3, 0)
        sc = np.percentile(b3, 75, 0) - np.percentile(b3, 25, 0) + 1e-9
        Z = (Xtr - mu) / sc
        Zte = (Xte - mu) / sc
        # PCA inside fold
        Zc = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        k = min(n_pca, Vt.shape[0])
        P = Vt[:k].T
        T = Zc @ P
        Tte = (Zte - Z.mean(0)) @ P
        df = pd.DataFrame(T, columns=[f"pc{j}" for j in range(k)])
        df["batch"] = ytr
        feats = df.columns[:k].tolist()
        model = CAT.build_model(df, feats)
        row = pd.Series(Tte[0], index=feats)
        d, best, conf, *_ = CAT.dist_and_predict(row, model)
        ds = [CAT.dist_and_predict(r, model)[0] for _, r in
              df[df.batch == C.BASELINE].iterrows()]
        thr = np.nanpercentile(ds, CAT.THRESH_Q * 100)
        preds.append(best); d_out.append(d); thr_l.append(thr)
    res = pd.DataFrame(dict(image_id=ids, batch=y, pred=preds,
                            dist=d_out, thr=thr_l))
    res["call_inout"] = np.where(res.dist <= res.thr, "IN", "OUT")
    return res


def group_acc(res_table_df, keys, embs, dets, n_pca, ns=(1, 3, 5, 7), trials=200):
    """Group-of-n accuracy per batch (median emb row per group)."""
    rng = np.random.default_rng(3)
    idx = [i for i, k in enumerate(keys) if k.split("|")[1] in dets]
    ids = sorted({keys[i].split("|")[0] for i in idx})
    X = np.stack([np.mean([embs[i] for i in idx
                           if keys[i].split("|")[0] == iid], 0)
                  for iid in ids])
    key_batch = dict(zip(res_table_df.image_id, res_table_df.batch))
    y = np.array([key_batch[i] for i in ids])
    b3 = X[y == C.BASELINE]
    mu = np.median(b3, 0); sc = np.percentile(b3, 75, 0) - \
        np.percentile(b3, 25, 0) + 1e-9
    Zc = (X - mu) / sc; Zc = Zc - Zc.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Vt[:n_pca].T
    T = Zc @ P
    df = pd.DataFrame(T, columns=[f"pc{j}" for j in range(n_pca)])
    df["batch"] = y; df["image_id"] = ids
    feats = df.columns[:n_pca].tolist()
    model = CAT.build_model(df, feats)
    out = []
    for b in ("Batch_1", "Batch_2"):
        idb = df[df.batch == b].index.tolist()
        for n in ns:
            hits = 0
            for _ in range(trials):
                pick = rng.choice(idb, size=min(n, len(idb)), replace=False)
                med = df.loc[pick, feats].median()
                _, best, *_ = CAT.dist_and_predict(med, model)
                hits += best == b
            out.append(dict(batch=b, n_images=n, group_acc=hits / trials))
    return pd.DataFrame(out)


def main():
    locs = C.find_locations()
    keys, embs = load_or_compute(locs)
    meta = {r["image_id"]: r["batch"] for r in locs}
    rows = []
    for dets, tag in [(("bse",), "bse"), (("inlens",), "inlens"),
                      (("topo",), "etd"), (("bse", "inlens", "topo"), "all")]:
        res = run_loo(meta, keys, embs, dets=dets)
        res["variant"] = tag
        acc = (res.batch == res.pred).mean()
        print(f"\n=== {tag}: LOO acc {acc:.3f} ===")
        print(pd.crosstab(res.batch, res.pred))
        fa = ((res.batch == C.BASELINE) & (res.call_inout == "OUT")).mean()
        print(f"false-alarm {fa:.2f}")
        g = group_acc(res, keys, embs, dets, N_PCA)
        print(g.pivot_table(index="n_images", columns="batch",
                            values="group_acc"))
        rows.append(res)
    pd.concat(rows).to_csv(RES_OUT, index=False)


if __name__ == "__main__":
    main()
