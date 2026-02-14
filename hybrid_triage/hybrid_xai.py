"""
Explainable AI (XAI) Module for Hybrid Triage System

Provides explanations for:
1. Rule-based layer: NEWS2 component score breakdowns
2. ML layer: SHAP-based feature importance and individual predictions
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../rule_based')
from rule_based_triage import calculate_news2_score

# XAI imports
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures


def explain_news2_scores(rule_df, indices, output_file="xai_outputs/xai_news2_explanations.txt"):
    """
    Explain rule-based predictions by showing NEWS2 component scores
    
    Args:
        rule_df: DataFrame with unscaled vitals
        indices: Indices of samples to explain
        output_file: Path to save explanations
    
    Returns:
        DataFrame with NEWS2 component breakdowns
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    explanations = []
    
    for idx in indices:
        row = rule_df.loc[idx]
        total_score, max_score, individual_scores = calculate_news2_score(row)
        
        explanations.append({
            'index': idx,
            'total_news2': total_score,
            'max_component': max_score,
            'resp_rate_score': individual_scores['resp_rate'],
            'o2sat_score': individual_scores['o2sat'],
            'temp_score': individual_scores['temperature'],
            'sbp_score': individual_scores['sbp'],
            'heart_rate_score': individual_scores['heart_rate'],
            'resp_rate_val': row.get('resprate', np.nan),
            'o2sat_val': row.get('o2sat', np.nan),
            'temp_val': row.get('temperature', np.nan),
            'sbp_val': row.get('sbp', np.nan),
            'heart_rate_val': row.get('heartrate', np.nan)
        })
    
    df_explanations = pd.DataFrame(explanations)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NEWS2 RULE-BASED EXPLANATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for _, exp in df_explanations.iterrows():
            f.write(f"Patient Index: {exp['index']}\n")
            f.write(f"Total NEWS2 Score: {exp['total_news2']}\n")
            f.write(f"Max Component Score: {exp['max_component']}\n")
            f.write("\nComponent Breakdown:\n")
            f.write(f"  Respiration Rate: {exp['resp_rate_val']:.1f} → score {exp['resp_rate_score']}\n")
            f.write(f"  O2 Saturation: {exp['o2sat_val']:.1f}% → score {exp['o2sat_score']}\n")
            f.write(f"  Temperature: {exp['temp_val']:.1f}°F → score {exp['temp_score']}\n")
            f.write(f"  Systolic BP: {exp['sbp_val']:.1f} → score {exp['sbp_score']}\n")
            f.write(f"  Heart Rate: {exp['heart_rate_val']:.1f} → score {exp['heart_rate_score']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"✓ Saved NEWS2 explanations to: {output_file}")
    return df_explanations


def generate_shap_explanations(stacking_model, X_ml, ml_indices, sample_size=100, 
                                output_dir="xai_outputs"):
    """
    Generate SHAP explanations for ML layer predictions
    
    Args:
        stacking_model: Trained stacking ensemble model
        X_ml: Feature data for ML predictions
        ml_indices: Indices of samples predicted by ML
        sample_size: Number of background samples for SHAP
        output_dir: Directory to save visualizations
    
    Returns:
        SHAP values, explainer, and feature importance DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating SHAP explanations for ML layer...")
    print(f"  ML samples to explain: {len(ml_indices)}")
    
    # Select background data for SHAP (use training data sample)
    X_background = X_ml.sample(n=min(sample_size, len(X_ml)), random_state=42)
    
    # Create SHAP explainer
    print(f"  Creating SHAP explainer with {len(X_background)} background samples...")
    explainer = shap.Explainer(stacking_model.predict_proba, X_background)
    
    # Calculate SHAP values for ML predictions
    X_explain = X_ml.loc[ml_indices]
    print(f"  Computing SHAP values for {len(X_explain)} samples...")
    shap_values = explainer(X_explain)
    
    # Generate summary plots for each class (predict_proba returns values for each class)
    class_names = ['Critical (Class 1)', 'Urgent (Class 2)', 'Non-Urgent (Class 3)']
    class_labels = [1, 2, 3]
    
    # Summary plot for each class
    print("  Creating SHAP summary plots for each class...")
    for class_idx in range(len(class_names)):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:,:,class_idx], X_explain, show=False, max_display=20)
        plt.title(f'SHAP Summary - {class_names[class_idx]}')
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"shap_summary_class_{class_labels[class_idx]}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {summary_path}")
    
    # Generate bar plots (mean absolute SHAP values) for each class
    print("  Creating SHAP feature importance bar plots for each class...")
    for class_idx in range(len(class_names)):
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values[:,:,class_idx], show=False, max_display=20)
        plt.title(f'SHAP Feature Importance - {class_names[class_idx]}')
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"shap_bar_class_{class_labels[class_idx]}.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {bar_path}")
    
    # Generate waterfall plots for first 5 samples (Critical class)
    print("  Creating individual waterfall plots (Critical - Class 1)...")
    for i in range(min(5, len(X_explain))):
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i,:,0], show=False)  # Class index 0 = Class 1 (Critical)
        plt.title(f'Sample {i} - Critical (Class 1) Probability')
        plt.tight_layout()
        waterfall_path = os.path.join(output_dir, f"shap_waterfall_sample_{i}_class_1.png")
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  ✓ Saved waterfall plots for first {min(5, len(X_explain))} samples")
    
    # Save SHAP values to CSV (for each class)
    for class_idx in range(len(class_names)):
        shap_df = pd.DataFrame(
            shap_values.values[:,:,class_idx],
            columns=[f"shap_{col}" for col in X_explain.columns],
            index=X_explain.index
        )
        shap_csv_path = os.path.join(output_dir, f"shap_values_class_{class_labels[class_idx]}.csv")
        shap_df.to_csv(shap_csv_path)
        print(f"  ✓ Saved SHAP values to: {shap_csv_path}")
    
    # Feature importance summary (average across all classes)
    mean_abs_shap_all_classes = np.abs(shap_values.values).mean(axis=(0, 2))  # Average across samples and classes
    feature_importance = pd.DataFrame({
        'feature': X_explain.columns,
        'mean_abs_shap': mean_abs_shap_all_classes
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"  ✓ Saved feature importance to: {importance_path}")
    
    print("\n✓ SHAP explanation generation complete!")
    
    return shap_values, explainer, feature_importance


def run_xai_analysis(stacking_model, rule_test, ml_test, source):
    """
    Run complete XAI analysis for hybrid triage system
    
    Args:
        stacking_model: Trained stacking ensemble model
        rule_test: Rule-based preprocessed test data
        ml_test: ML preprocessed test data
        source: Series indicating prediction source ('Rule' or 'ML')
    
    Returns:
        Dictionary with XAI results
    """
    print("\n" + "=" * 70)
    print("EXPLAINABLE AI (XAI) ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # 1. Explain rule-based predictions (NEWS2 breakdown)
    print("\n[1/2] Generating NEWS2 explanations for rule-based predictions...")
    rule_indices = source[source == 'Rule'].index[:50]  # First 50 rule-based samples
    if len(rule_indices) > 0:
        news2_explanations = explain_news2_scores(
            rule_test, 
            rule_indices, 
            output_file="xai_outputs/xai_news2_explanations.txt"
        )
        results['news2_explanations'] = news2_explanations
    else:
        print("  ⚠️  No rule-based predictions to explain")
        results['news2_explanations'] = None
    
    # 2. Generate SHAP explanations for ML layer
    print("\n[2/2] Generating SHAP explanations for ML layer predictions...")
    ml_indices = source[source == 'ML'].index[:500]  # First 500 ML-based samples
    if len(ml_indices) > 0:
        X_ml_full = ml_test.drop(columns=['acuity']) if 'acuity' in ml_test.columns else ml_test
        
        shap_values, explainer, feature_importance = generate_shap_explanations(
            stacking_model,
            X_ml_full,
            ml_indices,
            sample_size=100,
            output_dir="xai_outputs"
        )
        
        results['shap_values'] = shap_values
        results['explainer'] = explainer
        results['feature_importance'] = feature_importance
        
        # Display top features
        print("\n" + "=" * 70)
        print("TOP 10 MOST IMPORTANT FEATURES (ML Layer)")
        print("=" * 70)
        print(feature_importance.head(10).to_string(index=False))
    else:
        print("  ⚠️  No ML predictions to explain")
        results['shap_values'] = None
        results['explainer'] = None
        results['feature_importance'] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("XAI ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    if len(rule_indices) > 0:
        print("  - xai_outputs/xai_news2_explanations.txt")
    if len(ml_indices) > 0:
        print("  - xai_outputs/shap_summary_class_1.png, class_2.png, class_3.png")
        print("  - xai_outputs/shap_bar_class_1.png, class_2.png, class_3.png")
        print("  - xai_outputs/shap_waterfall_sample_*_class_1.png (up to 5 samples)")
        print("  - xai_outputs/shap_values_class_1.csv, class_2.csv, class_3.csv")
        print("  - xai_outputs/feature_importance.csv")
    
    return results


if __name__ == "__main__":
    print("XAI module for Hybrid Triage System")
    print("Import this module to use XAI functions:")
    print("  - explain_news2_scores()")
    print("  - generate_shap_explanations()")
    print("  - run_xai_analysis()")
