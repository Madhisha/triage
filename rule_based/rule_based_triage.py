"""
Rule-based triage layer using NEWS2 (National Early Warning Score 2) scoring system
Maps to 3 triage classes: Non-Urgent (0), Urgent (1), Emergency (2)
Merged into 3 classes, rule based prediction and evaluating it with ‘acuity’ column  in the dataset

"""

import pandas as pd
import numpy as np


def fahrenheit_to_celsius(temp_f):
    """Convert Fahrenheit to Celsius"""
    return (temp_f - 32) * 5/9


def get_respiration_score(resp_rate):
    """
    NEWS2 Respiration Rate scoring
    """
    if pd.isna(resp_rate):
        return 0
    
    if resp_rate <= 8:
        return 3
    elif resp_rate <= 11:
        return 1
    elif resp_rate <= 20:
        return 0
    elif resp_rate <= 24:
        return 2
    else:  # >= 25
        return 3


def get_o2sat_score(o2sat):
    """
    NEWS2 O2 Saturation scoring (Scale 1 - room air)
    """
    if pd.isna(o2sat):
        return 0
    
    if o2sat <= 91:
        return 3
    elif o2sat <= 93:
        return 2
    elif o2sat <= 95:
        return 1
    else:  # >= 96
        return 0


def get_temperature_score(temp_f):
    """
    NEWS2 Temperature scoring (converts F to C)
    Thresholds in Celsius:
    ≤35.0°C = 3
    35.1-36.0°C = 1
    36.1-38.0°C = 0
    38.1-39.0°C = 1
    ≥39.1°C = 2
    """
    if pd.isna(temp_f):
        return 0
    
    temp_c = fahrenheit_to_celsius(temp_f)
    
    if temp_c <= 35.0:
        return 3
    elif temp_c <= 36.0:
        return 1
    elif temp_c <= 38.0:
        return 0
    elif temp_c <= 39.0:
        return 1
    else:  # >= 39.1
        return 2


def get_sbp_score(sbp):
    """
    NEWS2 Systolic Blood Pressure scoring
    """
    if pd.isna(sbp):
        return 0
    
    if sbp <= 90:
        return 3
    elif sbp <= 100:
        return 2
    elif sbp <= 110:
        return 1
    elif sbp <= 219:
        return 0
    else:  # >= 220
        return 3


def get_heart_rate_score(hr):
    """
    NEWS2 Heart Rate scoring
    """
    if pd.isna(hr):
        return 0
    
    if hr <= 40:
        return 3
    elif hr <= 50:
        return 1
    elif hr <= 90:
        return 0
    elif hr <= 110:
        return 1
    elif hr <= 130:
        return 2
    else:  # >= 131
        return 3


def calculate_news2_score(row):
    """
    Calculate total NEWS2 score from vital signs
    Returns: (total_score, max_individual_score)
    """
    scores = {
        'resp_rate': get_respiration_score(row.get('resprate', np.nan)),
        'o2sat': get_o2sat_score(row.get('o2sat', np.nan)),
        'temperature': get_temperature_score(row.get('temperature', np.nan)),
        'sbp': get_sbp_score(row.get('sbp', np.nan)),
        'heart_rate': get_heart_rate_score(row.get('heartrate', np.nan))
    }
    
    total_score = sum(scores.values())
    max_score = max(scores.values())
    
    return total_score, max_score, scores


def rule_based_triage(row):
    """
    Apply NEWS2-based rule-based triage
    
    Mapping (Strict NEWS2 → 3 Classes):
    - NEWS2 ≥7 → Emergency (class 1)
    - NEWS2 5-6 OR any parameter = 3 → Urgent (class 2)
    - NEWS2 0-4 AND no parameter = 3 → Non-Urgent (class 3)
    
    Returns: triage class (1, 2, or 3) to match dataset acuity values
    """
    total_score, max_score, individual_scores = calculate_news2_score(row)
    
    # Emergency: total score >= 7 (class 1 - most urgent)
    if total_score >= 7:
        return 1
    
    # Urgent: total score 5-6 OR any single parameter score = 3 (class 2)
    elif total_score >= 5 or max_score >= 3:
        return 2
    
    # Non-urgent: total score 0-4 AND no parameter = 3 (class 3)
    else:
        return 3


def apply_rule_based_triage(df):
    """         
    Apply rule-based triage to preprocessed dataframe
    
    Args:
        df: Preprocessed DataFrame with vital signs (already cleaned)
    
    Returns:
        DataFrame with additional columns:
        - rule_based_prediction: triage class (1, 2, 3)
        - news2_total_score: total NEWS2 score
        - news2_max_score: max individual parameter score
    """
    df_copy = df.copy()
    
    # Calculate predictions and scores
    results = df_copy.apply(
        lambda row: pd.Series({
            'prediction': rule_based_triage(row),
            'total_score': calculate_news2_score(row)[0],
            'max_score': calculate_news2_score(row)[1]
        }), 
        axis=1
    )
    
    df_copy['rule_based_prediction'] = results['prediction']
    df_copy['news2_total_score'] = results['total_score']
    df_copy['news2_max_score'] = results['max_score']
    
    return df_copy


def merge_classes(y):
    """Merge classes 4, 5 -> 3 for 3-class problem"""
    return y.replace({4: 3, 5: 3})


def evaluate_rule_based_model(df, true_label_col='acuity'):
    """
    Evaluate rule-based model performance
    
    Args:
        df: DataFrame with rule_based_prediction and true labels
        true_label_col: name of column with true labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    df_copy = df.copy()
    
    # Merge classes 4, 5 -> 3
    df_copy[true_label_col] = merge_classes(df_copy[true_label_col])
    
    y_true = df_copy[true_label_col]
    y_pred = df_copy['rule_based_prediction']
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    class_report = classification_report(y_true, y_pred, labels=[1, 2, 3],
                                         target_names=['Emergency (1)', 'Urgent (2)', 'Non-Urgent (3)'])
    
    print("=" * 60)
    print("RULE-BASED TRIAGE MODEL EVALUATION (NEWS2)")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nClassification Report:")
    print(class_report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


# Example usage
if __name__ == "__main__":
    import os
    
    data_dir = "rule_processed_data"
    
    print("="*70)
    print("RULE-BASED TRIAGE EVALUATION (NEWS2)")
    print("Using preprocessed data (no duplicates, no missing values, clipped)")
    print("="*70)
    
    # Load preprocessed training data
    print("\nLoading preprocessed training data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'rule_train.csv'))
    print(f"Train shape: {train_df.shape}")
    
    # Apply rule-based triage
    print("\nApplying NEWS2 rule-based triage...")
    train_df_with_predictions = apply_rule_based_triage(train_df)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("TRAINING SET EVALUATION")
    print("=" * 70)
    metrics = evaluate_rule_based_model(train_df_with_predictions)
    
    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (TRAIN)")
    print("=" * 60)
    sample_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 
                   'news2_total_score', 'news2_max_score', 'rule_based_prediction', 'acuity']
    print(train_df_with_predictions[sample_cols].head(15))
    
    # Test on validation set
    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)
    valid_df = pd.read_csv(os.path.join(data_dir, 'rule_valid.csv'))
    print(f"Valid shape: {valid_df.shape}")
    valid_df_with_predictions = apply_rule_based_triage(valid_df)
    metrics_valid = evaluate_rule_based_model(valid_df_with_predictions)
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (VALID)")
    print("=" * 60)
    print(valid_df_with_predictions[sample_cols].head(15))
    
    # Test on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    test_df = pd.read_csv(os.path.join(data_dir, 'rule_test.csv'))
    print(f"Test shape: {test_df.shape}")
    test_df_with_predictions = apply_rule_based_triage(test_df)
    metrics_test = evaluate_rule_based_model(test_df_with_predictions)
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (TEST)")
    print("=" * 60)
    print(test_df_with_predictions[sample_cols].head(15))
    
    print("\n" + "="*70)
    print("✅ Rule-based triage evaluation complete!")
    print("="*70)
