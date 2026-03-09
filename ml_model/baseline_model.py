"""
Baseline Model for Patient Triage Prediction
=============================================
This script implements baseline models using DummyClassifier to establish
a performance floor for comparison with more sophisticated ML models.

Baseline strategies implemented:
1. Most Frequent: Always predicts the most common class
2. Stratified: Random predictions respecting training set class distribution
3. Uniform: Random predictions with equal probability for each class
4. Prior: Predicts class based on training set prior distribution

Class Configuration:
- 5-class problem: Acuity 1, 2, 3, 4, 5 (original labels)
- 3-class problem: 
    * Class 1: Critical (acuity 1)
    * Class 2: Urgent (acuity 2)
    * Class 3: Semi-urgent+ (acuity 3, 4, 5 merged)
"""

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import joblib
import os
from datetime import datetime


def load_data():
    """Load the preprocessed train, validation, and test datasets"""
    print("="*80)
    print("LOADING PREPROCESSED DATA")
    print("="*80)
    
    train_df = pd.read_csv("ml_processed_data/balanced/ml_processed_train.csv")
    valid_df = pd.read_csv("ml_processed_data/balanced/ml_processed_valid.csv")
    test_df = pd.read_csv("ml_processed_data/balanced/ml_processed_test.csv")
    
    print(f"\nDataset Shapes:")
    print(f"  Train: {train_df.shape}")
    print(f"  Valid: {valid_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    # Count feature types
    tfidf_cols = [col for col in train_df.columns if col.startswith('tfidf_')]
    numeric_cols = [col for col in train_df.columns if not col.startswith('tfidf_') and col != 'acuity']
    
    print(f"\nFeature Breakdown:")
    print(f"  - TF-IDF features (chief complaint): {len(tfidf_cols)}")
    print(f"  - Physiological features: {len(numeric_cols)}")
    print(f"  - Total features: {len(tfidf_cols) + len(numeric_cols)}")
    
    return train_df, valid_df, test_df


def prepare_features_target(df, target_col='acuity', merge_classes=False):
    """Separate features and target variable
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        merge_classes: If True, merge acuity 3,4,5 into single class 3
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # Merge classes 3, 4, 5 into class 3 for 3-class problem
    if merge_classes:
        y = y.replace({4.0: 3.0, 5.0: 3.0})
        y = y.replace({4: 3, 5: 3})  # Handle both int and float
        print("\nMerged classes 4 and 5 into class 3.")
        print(f"New class distribution: {sorted(y.unique())}")
    
    return X, y


def print_class_distribution(y, dataset_name):
    """Print class distribution statistics"""
    print(f"\n{dataset_name} Class Distribution:")
    print("-" * 60)
    
    class_counts = y.value_counts().sort_index()
    total = len(y)
    
    for acuity, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  Acuity {int(acuity)}: {count:>6,} ({percentage:>5.2f}%)")
    
    print(f"  Total:      {total:>6,}")
    print("-" * 60)


def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, strategy_name):
    """
    Evaluate baseline model on train, validation, and test sets
    Returns evaluation metrics for all datasets
    """
    print("\n" + "="*80)
    print(f"EVALUATING: {strategy_name}")
    print("="*80)
    
    results = {}
    
    for dataset_name, X, y in [('TRAIN', X_train, y_train), 
                                ('VALID', X_valid, y_valid), 
                                ('TEST', X_test, y_test)]:
        
        print(f"\n{'-'*80}")
        print(f"{dataset_name} SET EVALUATION")
        print(f"{'-'*80}")
        
        # Predictions
        y_pred = model.predict(X)
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f} (weighted)")
        print(f"  Recall:    {recall:.4f} (weighted)")
        print(f"  F1-Score:  {f1:.4f} (weighted)")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(classification_report(y, y_pred, digits=4))
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Calculate per-class accuracy from confusion matrix
        print(f"\nPer-Class Accuracy:")
        for i, acuity in enumerate(sorted(y.unique())):
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  Acuity {int(acuity)}: {class_accuracy:.4f}")
        
        # Try to compute AUC metrics if possible (not for most_frequent strategy)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                
                # Binarize labels for multi-class AUC
                y_bin = label_binarize(y, classes=sorted(y_train.unique()))
                
                # Compute AUROC (One-vs-Rest)
                auroc = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
                
                # Compute AUPRC (Average Precision)
                auprc = average_precision_score(y_bin, y_proba, average='weighted')
                
                print(f"\nAUROC (weighted): {auroc:.4f}")
                print(f"AUPRC (weighted): {auprc:.4f}")
                
                results[f'{dataset_name.lower()}_auroc'] = auroc
                results[f'{dataset_name.lower()}_auprc'] = auprc
        except:
            # Most frequent classifier doesn't support predict_proba
            pass
        
        # Store results
        results[f'{dataset_name.lower()}_accuracy'] = accuracy
        results[f'{dataset_name.lower()}_precision'] = precision
        results[f'{dataset_name.lower()}_recall'] = recall
        results[f'{dataset_name.lower()}_f1'] = f1
        results[f'{dataset_name.lower()}_confusion_matrix'] = cm
    
    return results


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BASELINE MODEL TRAINING AND EVALUATION")
    print("Patient Triage Prediction System")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    train_df, valid_df, test_df = load_data()
    
    # Ask user about class merging
    print("\n" + "="*80)
    print("CLASS CONFIGURATION")
    print("="*80)
    print("\nMerge acuity classes 3, 4, 5 into a single class?")
    print("  1. Yes - Merge to 3 classes (1: Critical, 2: Urgent, 3: Semi-urgent+)")
    print("  2. No  - Keep 5 classes (1, 2, 3, 4, 5)")
    
    merge_choice = input("\nEnter your choice (1 or 2): ").strip()
    merge_classes = merge_choice == '1'
    
    if merge_classes:
        print("\n✓ Classes will be merged: 3-class problem (1, 2, 3)")
    else:
        print("\n✓ Keeping original 5 classes (1, 2, 3, 4, 5)")
    
    # Prepare features and targets
    X_train, y_train = prepare_features_target(train_df, merge_classes=merge_classes)
    X_valid, y_valid = prepare_features_target(valid_df, merge_classes=merge_classes)
    X_test, y_test = prepare_features_target(test_df, merge_classes=merge_classes)
    
    # Print class distributions
    print_class_distribution(y_train, "Training Set")
    print_class_distribution(y_valid, "Validation Set")
    print_class_distribution(y_test, "Test Set")
    
    # Define baseline strategies to test
    strategies = [
        ('most_frequent', 'Most Frequent Class'),
        ('stratified', 'Stratified (Respects Class Distribution)'),
        ('uniform', 'Uniform Random'),
        ('prior', 'Prior Distribution (Constant)')
    ]
    
    # Store all results
    all_results = {}
    
    # Train and evaluate each baseline strategy
    for strategy, strategy_name in strategies:
        print(f"\n{'#'*80}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'#'*80}")
        
        # Initialize DummyClassifier with specific strategy
        model = DummyClassifier(strategy=strategy, random_state=42)
        
        # Train on training set
        print(f"\nTraining DummyClassifier with strategy='{strategy}'...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        # Evaluate on all datasets
        results = evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, strategy_name)
        
        # Store results
        all_results[strategy] = results
        
        # Save the model
        n_classes = len(y_train.unique())
        class_suffix = "_3class" if n_classes == 3 else "_5class"
        model_dir = "baseline_results"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"baseline_{strategy}{class_suffix}.pkl")
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Generate summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON - ALL BASELINE STRATEGIES")
    print("="*80)
    
    # Create summary dataframe
    summary_data = []
    for strategy, strategy_name in strategies:
        results = all_results[strategy]
        row = {
            'Strategy': strategy_name,
            'Train_Acc': results.get('train_accuracy', 0),
            'Valid_Acc': results.get('valid_accuracy', 0),
            'Test_Acc': results.get('test_accuracy', 0),
            'Train_F1': results.get('train_f1', 0),
            'Valid_F1': results.get('valid_f1', 0),
            'Test_F1': results.get('test_f1', 0),
        }
        
        # Add AUC metrics if available
        if 'test_auroc' in results:
            row['Test_AUROC'] = results['test_auroc']
            row['Test_AUPRC'] = results['test_auprc']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nAccuracy Comparison:")
    print(summary_df[['Strategy', 'Train_Acc', 'Valid_Acc', 'Test_Acc']].to_string(index=False))
    
    print("\nF1-Score Comparison:")
    print(summary_df[['Strategy', 'Train_F1', 'Valid_F1', 'Test_F1']].to_string(index=False))
    
    if 'Test_AUROC' in summary_df.columns:
        print("\nAUC Metrics (Test Set):")
        print(summary_df[['Strategy', 'Test_AUROC', 'Test_AUPRC']].to_string(index=False))
    
    # Save summary to file
    n_classes = len(y_train.unique())
    results_dir = "baseline_results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "baseline_results.txt")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODEL RESULTS SUMMARY\n")
        f.write(f"Problem Type: {n_classes}-class classification\n")
        f.write(f"Classes: {sorted(y_train.unique())}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Accuracy Comparison:\n")
        f.write(summary_df[['Strategy', 'Train_Acc', 'Valid_Acc', 'Test_Acc']].to_string(index=False))
        f.write("\n\nF1-Score Comparison:\n")
        f.write(summary_df[['Strategy', 'Train_F1', 'Valid_F1', 'Test_F1']].to_string(index=False))
        
        if 'Test_AUROC' in summary_df.columns:
            f.write("\n\nAUC Metrics (Test Set):\n")
            f.write(summary_df[['Strategy', 'Test_AUROC', 'Test_AUPRC']].to_string(index=False))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for strategy, strategy_name in strategies:
            f.write(f"\n{'='*80}\n")
            f.write(f"Strategy: {strategy_name}\n")
            f.write(f"{'='*80}\n\n")
            
            results = all_results[strategy]
            
            for dataset in ['train', 'valid', 'test']:
                f.write(f"\n{dataset.upper()} SET:\n")
                f.write(f"  Accuracy:  {results.get(f'{dataset}_accuracy', 0):.4f}\n")
                f.write(f"  Precision: {results.get(f'{dataset}_precision', 0):.4f}\n")
                f.write(f"  Recall:    {results.get(f'{dataset}_recall', 0):.4f}\n")
                f.write(f"  F1-Score:  {results.get(f'{dataset}_f1', 0):.4f}\n")
                
                if f'{dataset}_auroc' in results:
                    f.write(f"  AUROC:     {results[f'{dataset}_auroc']:.4f}\n")
                    f.write(f"  AUPRC:     {results[f'{dataset}_auprc']:.4f}\n")
                
                f.write(f"\nConfusion Matrix:\n")
                f.write(str(results[f'{dataset}_confusion_matrix']) + "\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("BASELINE MODEL EVALUATION COMPLETE")
    print(f"Problem Type: {n_classes}-class classification")
    print(f"Classes: {sorted(y_train.unique())}")
    print("="*80)
    
    # Final recommendations
    print("\nKey Insights:")
    print("  - Baseline models provide a performance floor for comparison")
    print("  - 'Most Frequent' strategy sets the minimum acceptable accuracy")
    print("  - Advanced ML models should significantly outperform these baselines")
    print("  - If your model doesn't beat these baselines, investigate feature engineering")
    if n_classes == 3:
        print("  - 3-class problem: Class 1=Critical, Class 2=Urgent, Class 3=Semi-urgent+")
    print("\n")


if __name__ == "__main__":
    # Make sure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
