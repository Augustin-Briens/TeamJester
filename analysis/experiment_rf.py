#!/usr/bin/env python3
"""RandomForest fusion classifier — honest evaluation.

Trains a RandomForest on the gated explainable feature set (the same kept
features categorise.py uses) for the 3-class batch call and the focused
B2-vs-B3 call, and reports:
  * leave-one-image-out accuracy + confusion (features/thresholds unchanged —
    the RF sees only the other 30 images each fold)
  * group-of-n accuracy (mean predicted probability over n images of a batch)
  * shuffled-label control
  * permutation importances (on held-out folds, aggregated) — the RF's own
    explainability output
Outputs land in outputs/tables/rf_*.csv and outputs/figures/rf_*.png.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, FIG_DIR as FIG, TABLE_DIR, BATCH_COLORS as BCOL
FEATURES_CSV = os.path.join(TABLE_DIR, "features.csv")

rng = np.random.default_rng(7)

full = pd.read_csv(FEATURES_CSV)
full = full[full.subset == "full"].reset_index(drop=True)
kept = pd.read_csv(os.path.join(TABLE_DIR, "baseline_stats.csv"))
kept_feats = [f for f in kept.feature if f in full.columns]
new = pd.read_csv(os.path.join(TABLE_DIR, "exp_newfeat_gates.csv"))
exp = pd.read_csv(os.path.join(TABLE_DIR, "exp_newfeat.csv"))
exp = exp[exp.subset == "full"].reset_index(drop=True)
new_feats = [f for f, ok in zip(new.feature, new.keep) if ok and f in exp.columns]
full = full.merge(exp[["image_id"] + new_feats], on="image_id", how="left")
FEATS = kept_feats + new_feats
print(f"{len(FEATS)} features, {len(full)} images")

X = full[FEATS].values.astype(float)
y = full["batch"].values
ids = full["image_id"].values
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]

def rf():
    return RandomForestClassifier(n_estimators=400, max_depth=4,
                                  min_samples_leaf=2, random_state=0,
                                  class_weight="balanced")

# ---- LOO -----------------------------------------------------------------
preds, probas, imp_acc = [], np.zeros((len(full), 3)), []
for i in range(len(full)):
    tr = np.arange(len(full)) != i
    m = rf().fit(X[tr], y[tr])
    preds.append(m.predict(X[i:i+1])[0])
    probas[i] = m.predict_proba(X[i:i+1])[0]
    pi = permutation_importance(m, X[tr], y[tr], n_repeats=15,
                                random_state=0, scoring="accuracy")
    imp_acc.append(pi.importances_mean)
preds = np.array(preds)
loo = (preds == y)
print("LOO acc:", loo.mean().round(3), {c: round(loo[y == c].mean(), 3) for c in CLASSES})

conf = pd.crosstab(pd.Series(y, name="true"), pd.Series(preds, name="pred"))
conf.to_csv(os.path.join(TABLE_DIR, "rf_confusion.csv"))
pd.DataFrame({"image_id": ids, "batch": y, "pred": preds,
              "correct": loo}).to_csv(os.path.join(TABLE_DIR, "rf_loo.csv"), index=False)

# ---- shuffled control -----------------------------------------------------
sh_acc = []
for rep in range(50):
    ys = rng.permutation(y)
    ok = 0
    for i in range(len(full)):
        tr = np.arange(len(full)) != i
        ok += rf().fit(X[tr], ys[tr]).predict(X[i:i+1])[0] == ys[i]
    sh_acc.append(ok / len(full))
print("shuffled:", np.mean(sh_acc).round(3))

# ---- group-of-n -----------------------------------------------------------
def group_acc(n, reps=300):
    acc = {c: [] for c in CLASSES}
    for c in CLASSES:
        idx = np.where(y == c)[0]
        for _ in range(reps):
            if len(idx) < n:
                acc[c].append(np.nan); continue
            sub = rng.choice(idx, n, replace=False)
            acc[c].append(CLASSES[np.argmax(probas[sub].mean(0))] == c)
    return {c: float(np.nanmean(v)) for c, v in acc.items()}

group_rows = []
for n in [1, 2, 3, 5, 7]:
    ga = group_acc(n)
    group_rows.append({"n": n, **{c: ga[c] for c in CLASSES}})
    print("group", n, {c: round(ga[c], 3) for c in CLASSES})
pd.DataFrame(group_rows).to_csv(os.path.join(TABLE_DIR, "rf_group_accuracy.csv"), index=False)

# ---- importances ----------------------------------------------------------
imp = np.mean(imp_acc, 0)
impdf = pd.DataFrame({"feature": FEATS, "perm_importance": imp}
                     ).sort_values("perm_importance", ascending=False)
impdf.to_csv(os.path.join(TABLE_DIR, "rf_importances.csv"), index=False)
print(impdf.head(10).to_string(index=False))

# ---- focused B2 vs B3 -----------------------------------------------------
sub = full[full.batch.isin(["Batch_2", "Batch_3"])].reset_index(drop=True)
Xs = sub[FEATS].values.astype(float); ys2 = sub["batch"].values
pr2 = []
for i in range(len(sub)):
    tr = np.arange(len(sub)) != i
    pr2.append(rf().fit(Xs[tr], ys2[tr]).predict(Xs[i:i+1])[0])
pr2 = np.array(pr2)
acc2 = (pr2 == ys2)
print("B2vsB3 RF LOO:", acc2.mean().round(3),
      "B2:", acc2[ys2 == "Batch_2"].mean().round(3))
pd.DataFrame({"image_id": sub.image_id, "batch": ys2, "pred": pr2, "correct": acc2}
             ).to_csv(os.path.join(TABLE_DIR, "rf_loo_b23.csv"), index=False)

# ---- tuned variant: top-8 features ---------------------------------------
m0 = RandomForestClassifier(400, max_depth=4, min_samples_leaf=2, random_state=0,
                            class_weight="balanced").fit(X, y)
TOP8 = [FEATS[i] for i in np.argsort(m0.feature_importances_)[::-1][:8]]
print("top8:", TOP8)
X8 = full[TOP8].values.astype(float)
preds8, probas8, imp8 = [], np.zeros((len(full), 3)), []
for i in range(len(full)):
    tr = np.arange(len(full)) != i
    m = rf().fit(X8[tr], y[tr])
    preds8.append(m.predict(X8[i:i+1])[0])
    probas8[i] = m.predict_proba(X8[i:i+1])[0]
    pi = permutation_importance(m, X8[tr], y[tr], n_repeats=15,
                                random_state=0, scoring="accuracy")
    imp8.append(pi.importances_mean)
preds8 = np.array(preds8)
loo8 = preds8 == y
print("tuned LOO:", loo8.mean().round(3), {c: round(loo8[y == c].mean(), 3) for c in CLASSES})
pd.DataFrame({"image_id": ids, "batch": y, "pred": preds8, "correct": loo8}
             ).to_csv(os.path.join(TABLE_DIR, "rf_loo_tuned.csv"), index=False)
np.savez(os.path.join(TABLE_DIR, "rf_probas.npz"), probas=probas8, y=y, ids=ids)

gt = []
rng2 = np.random.default_rng(1)
for n in [1, 2, 3, 5, 7]:
    row = {"n": n}
    for c in CLASSES:
        idx = np.where(y == c)[0]
        oks = [CLASSES[np.argmax(probas8[rng2.choice(idx, n, replace=False)].mean(0))] == c
               for _ in range(300)]
        row[c] = float(np.mean(oks))
    gt.append(row)
    print("tuned group", n, {c: round(row[c], 3) for c in CLASSES})
pd.DataFrame(gt).to_csv(os.path.join(TABLE_DIR, "rf_group_accuracy_tuned.csv"), index=False)

imp8df = pd.DataFrame({"feature": TOP8, "perm_importance": np.mean(imp8, 0)}
                      ).sort_values("perm_importance", ascending=False)
imp8df.to_csv(os.path.join(TABLE_DIR, "rf_importances_tuned.csv"), index=False)

# ---- figures --------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

top = imp8df.iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
cols = ["#0072B2" if f.startswith("etd") else "#E69F00" if f.startswith("bse")
        else "#009E73" for f in top.feature]
ax.barh(top.feature, top.perm_importance, color=cols)
for i, (f, v) in enumerate(zip(top.feature, top.perm_importance)):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("permutation importance (held-out accuracy lost when shuffled)")
ax.set_title("RandomForest (top-8) — what the votes rest on")
ax.annotate("surface texture + topography dominate;\nstructural features barely vote",
            xy=(top.perm_importance.iloc[-1] * 0.55, len(top) - 1.2),
            xytext=(top.perm_importance.iloc[-1] * 0.22, 2),
            arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9, color="grey")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rf_importances.png")); plt.close(fig)

gr = pd.DataFrame(group_rows); grt = pd.DataFrame(gt)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, dd, ttl in [(axes[0], gr, "all 28 features"), (axes[1], grt, "top-8 features")]:
    for c in CLASSES:
        ax.plot(dd.n, dd[c], "o-", color=BCOL[c], label=c)
    ax.axhline(1 / 3, color="grey", ls=":", lw=1)
    ax.annotate("chance", xy=(5.6, 0.345), fontsize=8, color="grey")
    ax.set_xlabel("images in group"); ax.set_title(ttl, fontsize=10)
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("fraction called correctly")
axes[1].legend(frameon=False, loc="lower right")
axes[0].annotate("collapses to Batch_3\n(B1, B2 near zero)",
                 xy=(5, 0.5), fontsize=8.5, color="#B22222",
                 xytext=(2.1, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#B22222"))
fig.suptitle("RandomForest group-of-n accuracy (mean vote) — honest LOO probabilities")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rf_group_accuracy.png")); plt.close(fig)
print("done")
