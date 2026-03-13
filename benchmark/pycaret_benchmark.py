"""
PyCaret Benchmark Script — Patient Triage Dataset
==================================================
Loads the already-processed data from ml_model/ml_processed_data/
and runs PyCaret AutoML to find the dataset's accuracy ceiling.

Why no SMOTE here?
  PyCaret's fix_imbalance=True applies SMOTE *within each CV fold*,
  which is the correct way (avoids data leakage). Pre-applying SMOTE
  would leak synthetic data into validation folds and inflate scores.

Usage:
    pip install pycaret
    cd benchmark
    python pycaret_benchmark.py
"""

import pandas as pd
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PYCARET BENCHMARK — Patient Triage Dataset")
    print("=" * 60)

    # --- Load already-processed data ---
    print("\n[1/3] Loading processed data...")

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
    train_df = train_df[train_df['acuity'].notna()]
    test_df = test_df[test_df['acuity'].notna()]
    train_df['acuity'] = train_df['acuity'].astype(int)
    test_df['acuity'] = test_df['acuity'].astype(int)

    print(f"  Merged to 3 classes (4,5 → 3)")
    print(f"  Train class distribution:\n{train_df['acuity'].value_counts().sort_index().to_string()}")

    # --- PyCaret setup ---
    print("\n[2/3] Setting up PyCaret...")

    try:
        from pycaret.classification import (
            setup, compare_models, predict_model,
            pull, save_model
        )
        
    except ImportError:
        print("ERROR: PyCaret not installed!")
        print("Install with: pip install pycaret")
        sys.exit(1)

    # Reset indices to avoid duplicate index errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    clf = setup(
        data=train_df,
        target='acuity',
        test_data=test_df,          # Use our existing test split
        index=False,                # Reset index to avoid duplicates
        session_id=42,
        normalize=False,            # Data is already scaled
        fix_imbalance=True,         # PyCaret applies SMOTE within CV folds
        fold=5,                     # 5-fold cross-validation
        verbose=False,
        html=False,
        log_experiment=False,
    )

    # --- Compare models ---
    print("\n[3/3] Comparing models (5-fold CV)...")
    print("  This may take 10-30 minutes...\n")

    best_models = compare_models(
        n_select=5,
        sort='Accuracy',
        turbo=True,
    )

    # Get results table
    results_df = pull()

    print("\n" + "=" * 60)
    print("PYCARET CV RESULTS (Training Data, 5-Fold)")
    print("=" * 60)
    print(results_df.to_string())

    # Save CV results
    results_path = os.path.join(SCRIPT_DIR, "pycaret_cv_results.csv")
    results_df.to_csv(results_path, index=True)
    print(f"\n✓ CV results saved to: {results_path}")

    # --- Evaluate best model on test set ---
    print("\n" + "=" * 60)
    print("BEST MODEL — TEST SET EVALUATION")
    print("=" * 60)

    best_model = best_models[0] if isinstance(best_models, list) else best_models
    model_name = type(best_model).__name__
    print(f"\nBest model: {model_name}")

    # Predict on test set
    test_predictions = predict_model(best_model, data=test_df)

    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score
    )

    y_test = test_predictions['acuity']
    y_pred = test_predictions['prediction_label']

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nTest Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Macro F1:  {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:")
    print(conf_matrix)

    # --- Append test results to the same CSV ---
    # Add a separator row and test set results
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write("\n\n# ============================================================\n")
        f.write("# TEST SET EVALUATION — Best Model\n")
        f.write("# ============================================================\n\n")

    # Save test summary as a row
    test_summary = pd.DataFrame([{
        'Model': f'{model_name} (TEST SET)',
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

    # Save comparison with existing model
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(f"\n# COMPARISON\n")
        f.write(f"# Your best (SMOTE + XGBoost): 70.03% accuracy, 0.67 macro F1\n")
        f.write(f"# PyCaret best ({model_name}): {test_acc*100:.2f}% accuracy, {test_f1:.2f} macro F1\n")

    print(f"\n✓ Test results appended to: {results_path}")

    # Save best model
    model_path = os.path.join(SCRIPT_DIR, "pycaret_best_model")
    save_model(best_model, model_path)
    print(f"\n✓ Best model saved to: {model_path}.pkl")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON: Your Model vs PyCaret Best")
    print("=" * 60)
    print(f"  Your best (SMOTE + XGBoost):     70.03% accuracy, 0.67 macro F1")
    print(f"  PyCaret best ({model_name}): {test_acc*100:.2f}% accuracy, {test_f1:.2f} macro F1")

    diff = test_acc * 100 - 70.03
    if diff > 1:
        print(f"\n  → PyCaret found {diff:.1f}% improvement! Worth investigating.")
    elif diff > 0:
        print(f"\n  → Only {diff:.1f}% improvement. Your model is near the ceiling.")
    else:
        print(f"\n  → Your current model is already at the ceiling! ✓")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
