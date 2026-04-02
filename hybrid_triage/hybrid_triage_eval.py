"""
Hybrid Triage System: Rule-Based (NEWS2) + Stacking Ensemble (LR)

Pipeline:
1. Apply rule-based NEWS2 triage to all samples
2. If classified as Critical (class 1) -> STOP, keep prediction
3. If NOT Critical -> pass to ML Stacking Ensemble (LR)
4. Combine predictions and evaluate on test set
5. Save results to text file
6. Generate XAI explanations (SHAP for ML, NEWS2 breakdown for rules)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import sys
sys.path.append('../rule_based')
from rule_based_triage import rule_based_triage

# Import XAI module
from hybrid_xai import run_xai_analysis


def merge_classes(y):
    """Merge classes 4, 5 -> 3 for 3-class problem"""
    return y.replace({4: 3, 5: 3})


def validate_rule_ml_alignment(rule_df, ml_df, min_match=0.99):
    """
    Validate that rule-based and ML test files are row-aligned.

    This script assigns rule-based routing from rule_df and ML predictions from
    ml_df by index. If row order differs, evaluation becomes invalid.
    """
    if len(rule_df) != len(ml_df):
        raise ValueError(
            f"Row count mismatch: rule_df={len(rule_df)}, ml_df={len(ml_df)}"
        )

    if 'acuity' not in rule_df.columns or 'acuity' not in ml_df.columns:
        raise ValueError("Both datasets must contain 'acuity' for alignment checks.")

    y_rule = merge_classes(rule_df['acuity']).reset_index(drop=True)
    y_ml = merge_classes(ml_df['acuity']).reset_index(drop=True)
    rowwise_match = (y_rule == y_ml).mean()

    print("\nAlignment check (rule_test vs ml_test):")
    print(f"  Row-wise merged-label match: {rowwise_match:.4f}")
    print(f"  Rule label distribution: {y_rule.value_counts().sort_index().to_dict()}")
    print(f"  ML label distribution  : {y_ml.value_counts().sort_index().to_dict()}")

    if rowwise_match < min_match:
        raise ValueError(
            "rule_test.csv and ml_processed_test.csv are not row-aligned. "
            f"Row-wise label match={rowwise_match:.4f} < {min_match:.2f}. "
            "Use the same patient-level split/order for both pipelines (or include "
            "a stable patient ID and join before evaluation)."
        )


def load_stacking_model(model_path):
    """Load pre-trained stacking ensemble model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded stacking ensemble from: {model_path}")
    return model


def apply_hybrid_triage(rule_df, ml_df, stacking_model):
    """
    Apply hybrid triage: Rule-based first, then ML for non-critical cases
    
    Args:
        rule_df: Preprocessed data for rule-based (unscaled vitals)
        ml_df: Preprocessed data for ML (scaled + features)
        stacking_model: Trained stacking ensemble model
    
    Returns:
        final_predictions: Combined predictions
        source: Array indicating prediction source ('Rule' or 'ML')
        rule_preds: Rule-based predictions for all samples
    """
    # Apply rule-based triage to all samples
    print("\nApplying NEWS2 rule-based triage...")
    rule_preds = rule_df.apply(lambda row: rule_based_triage(row), axis=1)

    # Determine how to map ML predictions into triage labels [1, 2, 3]
    model_classes = list(getattr(stacking_model, 'classes_', []))
    if model_classes == [0, 1, 2]:
        ml_label_shift = 1
    elif model_classes == [1, 2, 3]:
        ml_label_shift = 0
    else:
        raise ValueError(
            f"Unsupported stacking model classes: {model_classes}. "
            "Expected [0,1,2] or [1,2,3]."
        )
    
    # Identify critical cases (class 1)
    critical_mask = (rule_preds == 1)
    n_critical = critical_mask.sum()
    n_ml = (~critical_mask).sum()
    
    print(f"  Critical (class 1) by rules: {n_critical} ({n_critical/len(rule_preds)*100:.1f}%)")
    print(f"  Passed to ML: {n_ml} ({n_ml/len(rule_preds)*100:.1f}%)")
    
    # Prepare ML data (exclude acuity column)
    X_ml = ml_df.drop(columns=['acuity']) if 'acuity' in ml_df.columns else ml_df
    
    # Predict on non-critical samples using ML
    final_preds = pd.Series(0, index=rule_preds.index, dtype=int)
    source = pd.Series('Rule', index=rule_preds.index)
    
    # Keep rule-based predictions for critical cases
    final_preds[critical_mask] = 1
    
    # Use ML for non-critical cases
    if n_ml > 0:
        print(f"\nApplying ML (Stacking LR) to {n_ml} non-critical samples...")
        X_non_critical = X_ml[~critical_mask]
        ml_preds_raw = stacking_model.predict(X_non_critical)
        ml_preds = (ml_preds_raw + ml_label_shift).astype(int)
        final_preds[~critical_mask] = ml_preds
        source[~critical_mask] = 'ML'
    
    return final_preds, source, rule_preds


def evaluate_and_save_results(y_true, final_preds, source, rule_preds, output_file):
    """
    Evaluate hybrid model and save detailed results
    
    Args:
        y_true: True labels
        final_preds: Combined predictions
        source: Prediction source array
        rule_preds: Rule-based predictions
        output_file: Path to save results
    """
    labels = [1, 2, 3]
    target_names = ["Critical", "Urgent", "Non-Urgent"]
    
    # Overall metrics
    acc = accuracy_score(y_true, final_preds)
    macro_f1 = f1_score(y_true, final_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, final_preds, average='weighted', zero_division=0)
    macro_prec = precision_score(y_true, final_preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, final_preds, average='macro', zero_division=0)
    
    conf = confusion_matrix(y_true, final_preds, labels=labels)
    cls_report = classification_report(
        y_true, final_preds, labels=labels,
        target_names=target_names, zero_division=0
    )
    
    # Print to console
    print("\n" + "=" * 70)
    print("HYBRID MODEL RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy     : {acc:.4f}")
    print(f"Macro F1-Score       : {macro_f1:.4f}")
    print(f"Weighted F1-Score    : {weighted_f1:.4f}")
    print(f"Macro Precision      : {macro_prec:.4f}")
    print(f"Macro Recall         : {macro_rec:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'Crit':>8} {'Urg':>8} {'Non-Urg':>10}")
    for i, row in enumerate(conf):
        print(f"{target_names[i]:>12} {row[0]:>8} {row[1]:>8} {row[2]:>10}")
    
    print(f"\nClassification Report:")
    print(cls_report)
    
    # Breakdown by source
    print("=" * 70)
    print("BREAKDOWN BY PREDICTION SOURCE")
    print("=" * 70)
    
    n_rule = (source == 'Rule').sum()
    n_ml = (source == 'ML').sum()
    
    # Critical patient analysis
    total_critical = (y_true == 1).sum()
    critical_by_rule = ((y_true == 1) & (source == 'Rule') & (final_preds == 1)).sum()
    critical_by_ml = ((y_true == 1) & (source == 'ML') & (final_preds == 1)).sum()
    
    print(f"\nCRITICAL PATIENT CLASSIFICATION:")
    print(f"  Total true critical patients: {total_critical}")
    print(f"  Correctly classified by Rule-Based layer: {critical_by_rule} ({critical_by_rule/total_critical*100:.1f}%)")
    print(f"  Correctly classified by ML layer: {critical_by_ml} ({critical_by_ml/total_critical*100:.1f}%)")
    print(f"  Total correctly classified (combined): {critical_by_rule + critical_by_ml} ({(critical_by_rule + critical_by_ml)/total_critical*100:.1f}%)")
    
    if n_rule > 0:
        rule_acc = accuracy_score(y_true[source == 'Rule'], 
                                  final_preds[source == 'Rule'])
        print(f"\nRule-Based subset ({n_rule} samples):")
        print(f"  Accuracy: {rule_acc:.4f}")
    
    if n_ml > 0:
        ml_acc = accuracy_score(y_true[source == 'ML'], 
                               final_preds[source == 'ML'])
        ml_f1 = f1_score(y_true[source == 'ML'], 
                        final_preds[source == 'ML'], 
                        average='macro', zero_division=0)
        print(f"\nML (Stacking LR) subset ({n_ml} samples):")
        print(f"  Accuracy: {ml_acc:.4f}")
        print(f"  Macro F1: {ml_f1:.4f}")
    
    # Save to file
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    lines = [
        "=" * 70,
        "HYBRID TRIAGE SYSTEM: Rule-Based (NEWS2) + Stacking Ensemble (LR)",
        f"Evaluated on: Test Set",
        "=" * 70,
        "",
        "PIPELINE DESCRIPTION:",
        "  1. Apply rule-based NEWS2 triage to all samples",
        "  2. If classified as Critical (class 1) -> STOP, keep prediction",
        "  3. If NOT Critical -> pass to ML Stacking Ensemble (LR)",
        "  4. Base models in stacking: Random Forest, XGBoost, MLP",
        "  5. Meta-learner: Logistic Regression",
        "",
        "-" * 70,
        "DATA SUMMARY:",
        f"  Test samples: {len(y_true)}",
        f"  Classes: {sorted(y_true.unique().tolist())}",
        "",
        "  True label distribution:",
    ]
    
    for cls in sorted(y_true.unique()):
        count = (y_true == cls).sum()
        lines.append(f"    Class {int(cls)} ({target_names[int(cls)-1]:>11}): {count:>6} ({count/len(y_true)*100:.1f}%)")
    
    lines.extend([
        "",
        "-" * 70,
        "ROUTING SUMMARY:",
        f"  Rule-based -> Critical (class 1): {n_rule:>6} ({n_rule/len(y_true)*100:.1f}%)",
        f"  Passed to ML (Stacking LR)      : {n_ml:>6} ({n_ml/len(y_true)*100:.1f}%)",
        "",
        "-" * 70,
        "OVERALL RESULTS:",
        f"  Accuracy         : {acc:.4f}",
        f"  Macro F1-Score   : {macro_f1:.4f}",
        f"  Weighted F1-Score: {weighted_f1:.4f}",
        f"  Macro Precision  : {macro_prec:.4f}",
        f"  Macro Recall     : {macro_rec:.4f}",
        "",
        "-" * 70,
        "CONFUSION MATRIX (rows=true, cols=predicted):",
        f"  {'':>12} {'Critical':>10} {'Urgent':>10} {'Non-Urgent':>12}",
    ])
    
    for i, row in enumerate(conf):
        lines.append(f"  {target_names[i]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>12}")
    
    lines.extend([
        "",
        "-" * 70,
        "CLASSIFICATION REPORT:",
        cls_report,
        "-" * 70,
        "BREAKDOWN BY PREDICTION SOURCE:",
        "",
        "CRITICAL PATIENT CLASSIFICATION:",
        f"  Total true critical patients: {total_critical}",
        f"  Correctly classified by Rule-Based layer: {critical_by_rule} ({critical_by_rule/total_critical*100:.1f}%)",
        f"  Correctly classified by ML layer: {critical_by_ml} ({critical_by_ml/total_critical*100:.1f}%)",
        f"  Total correctly classified (combined): {critical_by_rule + critical_by_ml} ({(critical_by_rule + critical_by_ml)/total_critical*100:.1f}%)",
    ])
    
    if n_rule > 0:
        lines.extend([
            "",
            f"Rule-Based subset ({n_rule} samples):",
            f"  Accuracy: {rule_acc:.4f}",
            f"  True class distribution:",
        ])
        for cls in sorted(y_true[source == 'Rule'].unique()):
            count = (y_true[source == 'Rule'] == cls).sum()
            lines.append(f"    Class {int(cls)} ({target_names[int(cls)-1]}): {count}")
    
    if n_ml > 0:
        ml_report = classification_report(
            y_true[source == 'ML'], 
            final_preds[source == 'ML'],
            labels=labels, target_names=target_names, zero_division=0
        )
        lines.extend([
            "",
            f"ML (Stacking LR) subset ({n_ml} samples):",
            f"  Accuracy: {ml_acc:.4f}",
            f"  Macro F1: {ml_f1:.4f}",
            "",
            "  Classification Report:",
            ml_report,
        ])
    
    lines.append("=" * 70)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Results saved to: {output_file}")


def main():
    print("=" * 70)
    print("HYBRID TRIAGE SYSTEM EVALUATION")
    print("Rule-Based (NEWS2) + Stacking Ensemble (LR)")
    print("=" * 70)
    print()
    
    # Paths
    rule_data_path = "../rule_based/rule_processed_data/rule_test.csv"
    ml_data_path = "../ml_model/ml_processed_data/ml_processed_test.csv"
    model_path = "../ml_model/ensemble_model/stacking_lr_ensemble.pkl"
    output_file = "hybrid_triage_results.txt"
    
    # Load data
    print("Loading test data...")
    print(f"  Rule-based data: {rule_data_path}")
    print(f"  ML data: {ml_data_path}")
    
    rule_test = pd.read_csv(rule_data_path)
    ml_test = pd.read_csv(ml_data_path)
    
    print(f"  Rule test shape: {rule_test.shape}")
    print(f"  ML test shape: {ml_test.shape}")

    # Guard against invalid hybrid evaluation due to row-order mismatch.
    validate_rule_ml_alignment(rule_test, ml_test)
    
    # Extract true labels and merge classes
    y_true = merge_classes(rule_test['acuity'])
    
    # Load stacking model
    print(f"\nLoading stacking ensemble model...")
    print(f"  Model path: {model_path}")
    stacking_model = load_stacking_model(model_path)
    
    # Check model's class labels
    print(f"\n🔍 IMPORTANT: Checking stacking model classes...")
    print(f"  Stacking model classes: {stacking_model.classes_}")
    if list(stacking_model.classes_) == [0, 1, 2]:
        print("  ✓ Model uses 0-indexed classes [0,1,2] → will add +1 to map to [1,2,3]")
    elif list(stacking_model.classes_) == [1, 2, 3]:
        print("  ⚠️  WARNING: Model already uses [1,2,3] → should NOT add +1!")
    else:
        print(f"  ⚠️  UNEXPECTED classes: {stacking_model.classes_}")
    
    # Apply hybrid triage
    print("\n" + "=" * 70)
    print("APPLYING HYBRID TRIAGE")
    print("=" * 70)
    
    final_preds, source, rule_preds = apply_hybrid_triage(
        rule_test, ml_test, stacking_model
    )
    
    # Evaluate and save results
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    eval_stats = evaluate_and_save_results(y_true, final_preds, source, rule_preds, output_file)
    
    # Run XAI analysis using separate module
    xai_results = run_xai_analysis(stacking_model, rule_test, ml_test, source)
    
    print("\n" + "=" * 70)
    print("✅ HYBRID TRIAGE EVALUATION + XAI COMPLETE!")
    print("=" * 70)
    print("\nMain results file:")
    print(f"  - {output_file}")
    print("\nXAI outputs: See above for detailed file list")


if __name__ == "__main__":
    main()
