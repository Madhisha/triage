import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(ROOT, "rule_based"))
from rule_based_triage import rule_based_triage


def main():
    rule_path = os.path.join(ROOT, "rule_based", "rule_processed_data", "rule_test.csv")
    ml_path = os.path.join(ROOT, "ml_model", "ml_processed_data", "ml_processed_test.csv")
    model_path = os.path.join(ROOT, "ml_model", "ensemble_model", "stacking_lr_ensemble.pkl")

    rule_df = pd.read_csv(rule_path)
    ml_df = pd.read_csv(ml_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Merge to 3 classes (4,5 -> 3)
    y_true_rule = rule_df["acuity"].replace({4: 3, 5: 3}).astype(int)
    y_true_ml = ml_df["acuity"].replace({4: 3, 5: 3}).astype(int)

    row_match = float((y_true_rule.reset_index(drop=True) == y_true_ml.reset_index(drop=True)).mean())
    if row_match < 0.99:
        raise ValueError(f"Row alignment issue: merged label row-wise match={row_match:.4f}")

    X_ml = ml_df.drop(columns=["acuity"], errors="ignore")

    # Ensemble probabilities, remapped to class labels [1,2,3]
    proba_raw = model.predict_proba(X_ml)
    model_classes = np.array(getattr(model, "classes_", []), dtype=int)

    if model_classes.tolist() == [0, 1, 2]:
        mapped_classes = model_classes + 1
    elif model_classes.tolist() == [1, 2, 3]:
        mapped_classes = model_classes
    else:
        raise ValueError(f"Unsupported model classes: {model_classes.tolist()}")

    proba_df = pd.DataFrame(proba_raw, columns=mapped_classes)
    proba_ensemble = proba_df.reindex(columns=[1, 2, 3], fill_value=0.0).to_numpy()

    # Rule-based deterministic scores (one-hot from hard class predictions)
    # Note: rule system does not output calibrated probabilities.
    rule_preds = rule_df.apply(lambda row: rule_based_triage(row), axis=1).astype(int)
    proba_rule = np.zeros((len(rule_preds), 3), dtype=float)
    for i, cls in enumerate([1, 2, 3]):
        proba_rule[:, i] = (rule_preds.to_numpy() == cls).astype(float)

    # Hybrid probabilities: force critical rule-based route to class 1 with prob=1
    rule_preds = rule_df.apply(lambda row: rule_based_triage(row), axis=1).astype(int)
    critical_mask = (rule_preds == 1).to_numpy()

    proba_hybrid = proba_ensemble.copy()
    proba_hybrid[critical_mask] = np.array([1.0, 0.0, 0.0])

    classes = np.array([1, 2, 3])
    y_bin = label_binarize(y_true_rule, classes=classes)

    auroc_ensemble = roc_auc_score(y_bin, proba_ensemble, average=None, multi_class="ovr")
    auroc_rule = roc_auc_score(y_bin, proba_rule, average=None, multi_class="ovr")
    auroc_hybrid = roc_auc_score(y_bin, proba_hybrid, average=None, multi_class="ovr")

    print("=" * 70)
    print("PER-CLASS AUROC (OVR)")
    print("=" * 70)
    print(f"Rows checked: {len(y_true_rule)} | rule/ml label row-match: {row_match:.4f}")
    print(f"Stacking model classes_: {model_classes.tolist()} -> mapped to {mapped_classes.tolist()}")
    print()
    print("Ensemble (Stacking LR) AUROC by class:")
    for c, v in zip(classes, auroc_ensemble):
        print(f"  Class {c}: {v:.6f}")
    print()
    print("Rule-Based (NEWS2) AUROC by class:")
    for c, v in zip(classes, auroc_rule):
        print(f"  Class {c}: {v:.6f}")
    print()
    print("Hybrid (Rule + Stacking) AUROC by class:")
    for c, v in zip(classes, auroc_hybrid):
        print(f"  Class {c}: {v:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
