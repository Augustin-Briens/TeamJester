"""Score the human-filled point-count validation sheet.

    python3 score_pointcount.py [path/to/point_count_validation.xlsx]

Reads the sheet's `human_class` column (pore | graphite | silicon), compares
against the predicted class, and reports overall accuracy, per-class recall
and the confusion matrix. The human column MUST be filled by a person;
missing rows are reported and skipped.
"""
import os
import sys
import pandas as pd
import common as C

CLASSES = ["pore", "graphite", "silicon"]


def main(path=None):
    path = path or os.path.join(C.PC_DIR, "point_count_validation.xlsx")
    df = pd.read_excel(path)
    if "human_class" not in df.columns:
        raise SystemExit("no human_class column in sheet")
    df["human_class"] = df["human_class"].astype(str).str.strip().str.lower()
    filled = df[df.human_class.isin(CLASSES)]
    print(f"{len(filled)} of {len(df)} points have a human label "
          f"({len(df)-len(filled)} skipped)")
    if not len(filled):
        return
    acc = (filled.human_class == filled.predicted_class).mean()
    print(f"overall accuracy: {acc:.3f}")
    rec = filled.groupby("human_class").apply(
        lambda g: (g.human_class == g.predicted_class).mean())
    print("per-class recall:")
    print(rec.to_string())
    cm = pd.crosstab(filled.human_class, filled.predicted_class)
    cm = cm.reindex(index=CLASSES, columns=CLASSES).fillna(0).astype(int)
    print("\nconfusion matrix (rows=human, cols=predicted):")
    print(cm.to_string())
    out = pd.DataFrame({"metric": ["n_labeled", "n_total", "accuracy"],
                        "value": [len(filled), len(df), acc]})
    outp = os.path.join(C.TABLE_DIR, "pointcount_scores.csv")
    out.to_csv(outp, index=False)
    cm.to_csv(os.path.join(C.TABLE_DIR, "pointcount_confusion.csv"))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
