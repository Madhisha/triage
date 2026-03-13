"""
AutoGluon Benchmark Script — Patient Triage Dataset
====================================================
Uses Amazon's AutoGluon AutoML to find the dataset's accuracy ceiling.
AutoGluon automatically builds multi-layer stacking ensembles and can
often squeeze out 1-2% more than single models.

Usage:
    pip install autogluon
    cd benchmark
    python autogluon_benchmark.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'ml_model', 'ml_processed_data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'autogluon_models')


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("AUTOGLUON BENCHMARK — Patient Triage Dataset")
    print("=" * 60)

    # --- Load already-processed data ---
    print("\n[1/4] Loading processed data...")

    train_path = os.path.join(PROCESSED_DIR, "ml_processed_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "ml_processed_test.csv")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found!")
        print("Run 'cd ml_model && python ml_preprocess.py' first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
    print(f"  Test:  {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")

    # Merge into 3 classes: 1=Critical, 2=Urgent, 3=Non-Urgent (4,5 → 3)
    for df in [train_df, test_df]:
        df['acuity'] = df['acuity'].map({1.0: 1, 2.0: 2, 3.0: 3, 4.0: 3, 5.0: 3,
                                          1: 1, 2: 2, 3: 3, 4: 3, 5: 3})
    train_df = train_df[train_df['acuity'].notna()].reset_index(drop=True)
    test_df = test_df[test_df['acuity'].notna()].reset_index(drop=True)
    train_df['acuity'] = train_df['acuity'].astype(int)
    test_df['acuity'] = test_df['acuity'].astype(int)

    print(f"  Merged to 3 classes (4,5 → 3)")
    print(f"  Train class distribution:\n{train_df['acuity'].value_counts().sort_index().to_string()}")

    # --- AutoGluon setup ---
    print("\n[2/4] Setting up AutoGluon...")

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("ERROR: AutoGluon not installed!")
        print("Install with: pip install autogluon")
        print("  or minimal: pip install autogluon.tabular")
        sys.exit(1)

    # --- Train ---
    print("\n[3/4] Training AutoGluon models...")
    print("  AutoGluon will try multiple models + stacking.")
    print("  This may take 15-60 minutes depending on your machine.\n")

    # Time limit in seconds (adjust as needed)
    # 600 = 10 minutes, 1800 = 30 minutes, 3600 = 1 hour
    TIME_LIMIT = 600  # 10 minutes

    predictor = TabularPredictor(
        label='acuity',
        path=OUTPUT_DIR,
        eval_metric='accuracy',         # Primary metric
        problem_type='multiclass',
    ).fit(
        train_data=train_df,
        time_limit=TIME_LIMIT,
        presets='best_quality',          # Uses stacking + bagging for best results
        verbosity=2,
    )

    # --- Evaluate ---
    print("\n[4/4] Evaluating on test set...")

    # Leaderboard (all models ranked)
    leaderboard = predictor.leaderboard(test_df, silent=True)

    print("\n" + "=" * 60)
    print("AUTOGLUON LEADERBOARD (Test Set)")
    print("=" * 60)
    print(leaderboard.to_string())

    # Save leaderboard
    results_path = os.path.join(SCRIPT_DIR, "autogluon_results.csv")
    leaderboard.to_csv(results_path, index=False)
    print(f"\n✓ Leaderboard saved to: {results_path}")

    # Best model evaluation
    print("\n" + "=" * 60)
    print("BEST MODEL — DETAILED TEST EVALUATION")
    print("=" * 60)

    y_test = test_df['acuity']
    y_pred = predictor.predict(test_df.drop(columns=['acuity']))

    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score
    )

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    best_model_name = predictor.model_best
    print(f"\nBest model: {best_model_name}")
    print(f"\nTest Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Macro F1:  {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:")
    print(conf_matrix)

    # --- Append test results to CSV ---
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\n\n# ============================================================\n")
        f.write("# TEST SET EVALUATION — Best Model\n")
        f.write("# ============================================================\n\n")

    test_summary = pd.DataFrame([{
        'Model': f'{best_model_name} (TEST SET)',
        'Accuracy': test_acc,
        'Precision': test_precision,
        'Recall': test_recall,
        'Macro F1': test_f1,
        'Class 1 Precision': cls_report['1']['precision'],
        'Class 1 Recall': cls_report['1']['recall'],
        'Class 1 F1': cls_report['1']['f1-score'],
        'Class 2 Precision': cls_report['2']['precision'],
        'Class 2 Recall': cls_report['2']['recall'],
        'Class 2 F1': cls_report['2']['f1-score'],
        'Class 3 Precision': cls_report['3']['precision'],
        'Class 3 Recall': cls_report['3']['recall'],
        'Class 3 F1': cls_report['3']['f1-score'],
    }])
    test_summary.to_csv(results_path, mode='a', index=False)

    # Save confusion matrix
    conf_df = pd.DataFrame(
        conf_matrix,
        index=['Actual_1', 'Actual_2', 'Actual_3'],
        columns=['Pred_1', 'Pred_2', 'Pred_3']
    )
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\n# Confusion Matrix (Test Set)\n")
    conf_df.to_csv(results_path, mode='a')

    # --- Summary comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON: Your Model vs AutoGluon Best")
    print("=" * 60)
    print(f"  Your best (SMOTE + XGBoost):     70.03% accuracy, 0.67 macro F1")
    # print(f"  PyCaret best (CatBoost):          70.13% accuracy, 0.67 macro F1")
    print(f"  AutoGluon best ({best_model_name}): {test_acc*100:.2f}% accuracy, {test_f1:.2f} macro F1")

    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(f"\n# COMPARISON\n")
        f.write(f"# Your best (SMOTE + XGBoost): 70.03% accuracy, 0.67 macro F1\n")
        # f.write(f"# PyCaret best (CatBoost): 70.13% accuracy, 0.67 macro F1\n")
        f.write(f"# AutoGluon best ({best_model_name}): {test_acc*100:.2f}% accuracy, {test_f1:.2f} macro F1\n")

    print(f"\n✓ All results saved to: {results_path}")

    diff = test_acc * 100 - 70.03
    if diff > 1:
        print(f"\n  → AutoGluon found {diff:.1f}% improvement! Worth investigating.")
    elif diff > 0:
        print(f"\n  → Only {diff:.1f}% improvement. Dataset is the bottleneck.")
    else:
        print(f"\n  → Your current model is already at the ceiling! ✓")

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 60)
    try:
        importance = predictor.feature_importance(test_df)
        print(importance.head(20).to_string())
        importance.to_csv(os.path.join(SCRIPT_DIR, "autogluon_feature_importance.csv"))
        print(f"\n✓ Feature importance saved to: autogluon_feature_importance.csv")
    except Exception as e:
        print(f"  Could not compute feature importance: {e}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
