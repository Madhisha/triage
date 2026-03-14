import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import joblib
import os

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from tuning.tune_random_forest import train_random_forest, tune_random_forest_random, tune_random_forest_grid, tune_random_forest_bayesian
from tuning.tune_xgboost import train_xgboost, tune_xgboost_random, tune_xgboost_grid, tune_xgboost_bayesian
from tuning.tune_mlp import train_mlp, tune_mlp_random, tune_mlp_grid, tune_mlp_bayesian
from tuning.tune_lightgbm import train_lightgbm, tune_lightgbm_random, tune_lightgbm_grid, tune_lightgbm_bayesian
from tuning.tune_catboost import train_catboost, tune_catboost_random, tune_catboost_grid, tune_catboost_bayesian
from tuning.tune_adaboost import train_adaboost, tune_adaboost_random, tune_adaboost_grid, tune_adaboost_bayesian
from tuning.tune_logistic_regression import train_logistic_regression, tune_logistic_regression_random, tune_logistic_regression_grid, tune_logistic_regression_bayesian
from tuning.tune_svm import train_svm, tune_svm_random, tune_svm_grid, tune_svm_bayesian

def get_model_params(model):
    """Return a shallow parameter snapshot for reporting."""
    if hasattr(model, 'get_params'):
        return model.get_params(deep=False)
    return {}

def attach_training_summary(model, model_name, strategy, best_validation_accuracy=None, tuning_results=None):
    """Attach training metadata to a model instance for report generation."""
    model.training_summary = {
        'model_name': model_name,
        'strategy': strategy,
        'best_validation_accuracy': best_validation_accuracy,
        'tuning_results': tuning_results or [],
        'params': get_model_params(model)
    }
    return model

def write_training_summary(output_file, model):
    """Write model training and tuning summary to the results file."""
    if not output_file or not hasattr(model, 'training_summary'):
        return

    summary = model.training_summary
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Training Summary ({summary['model_name']})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Selected strategy: {summary['strategy']}\n")

        if summary['best_validation_accuracy'] is not None:
            f.write(f"Best validation accuracy: {summary['best_validation_accuracy']:.4f}\n")

        if summary['tuning_results']:
            f.write("\nTuning method validation accuracies:\n")
            for result in summary['tuning_results']:
                f.write(f"- {result['method']}: {result['validation_accuracy']:.4f}\n")
                f.write(f"  Params: {result['params']}\n")

        if summary['params']:
            f.write("\nFinal model parameters:\n")
            for key in sorted(summary['params']):
                f.write(f"- {key}: {summary['params'][key]}\n")

def load_data():
    """Load the preprocessed train, validation, and test datasets with chief complaint features"""
    # train_df = pd.read_csv("ml_processed_data/ml_processed_train.csv")
    # valid_df = pd.read_csv("ml_processed_data/ml_processed_valid.csv")
    # test_df = pd.read_csv("ml_processed_data/ml_processed_test.csv")

    train_df = pd.read_csv("ml_processed_data/balanced/ml_processed_train.csv")
    valid_df = pd.read_csv("ml_processed_data/balanced/ml_processed_valid.csv")
    test_df = pd.read_csv("ml_processed_data/balanced/ml_processed_test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {valid_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Count feature types
    tfidf_cols = [col for col in train_df.columns if col.startswith('tfidf_')]
    numeric_cols = [col for col in train_df.columns if not col.startswith('tfidf_') and col != 'acuity']
    
    print(f"\nFeature breakdown:")
    print(f"  - TF-IDF (Chief Complaint) features: {len(tfidf_cols)}")
    print(f"  - Physiological features: {len(numeric_cols)}")
    print(f"  - Total features: {len(tfidf_cols) + len(numeric_cols)}")
    
    return train_df, valid_df, test_df

def prepare_features_target(df, target_col='acuity', merge_classes=False):
    """Separate features and target variable"""
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # Merge classes 3, 4, 5 into class 3 (high acuity)
    if merge_classes:
        y = y.replace({4.0: 3.0, 5.0: 3.0})
        print("Merged classes 4 and 5 into class 3.")
        print(f"New class distribution:\n{y.value_counts().sort_index()}")
    
    return X, y


# Imports from modular tuning package already handled at top of file


# ==================== XGBoost Tuning ====================


# ==================== MLP Tuning ====================


# ==================== LightGBM Tuning ====================


# ==================== CatBoost Tuning ====================


# ==================== AdaBoost Tuning ====================


# ==================== Logistic Regression Tuning ====================


# ==================== SVM Tuning ====================


def evaluate_model(model, X, y, dataset_name="Dataset", is_xgb=False, is_catboost=False, is_lightgbm=False, output_file=None):
    """Evaluate model performance with accuracy, classification report, AUROC, and AUPRC"""
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_name}")
    print("="*60)
    
    y_pred = model.predict(X)
    
    # Convert predictions back to original labels (0,1,2 -> 1,2,3)
    if is_xgb or is_catboost or is_lightgbm:
        y_pred = y_pred + 1
    
    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compute AUROC and AUPRC
    try:
        # Get probability predictions
        y_proba = model.predict_proba(X)
        
        # Adjust probabilities for XGBoost/CatBoost/LightGBM (0-indexed)
        if is_xgb or is_catboost or is_lightgbm:
            # For 0-indexed models, probabilities are already aligned with classes 0,1,2
            # But we need to align them with true labels 1,2,3
            classes = np.array([1, 2, 3])  # True class labels
        else:
            classes = model.classes_
        
        # Binarize the labels for multi-class ROC/PR computation
        y_bin = label_binarize(y, classes=classes)
        
        # Handle binary classification case (returns 1D array)
        if len(classes) == 2:
            y_bin = np.hstack([1 - y_proba[:, 1:2], y_proba[:, 1:2]])
        
        # Compute macro-averaged AUROC and AUPRC
        auroc_macro = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
        auprc_macro = average_precision_score(y_bin, y_proba, average='macro')
        
        # Compute weighted-averaged AUROC and AUPRC
        auroc_weighted = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        auprc_weighted = average_precision_score(y_bin, y_proba, average='weighted')
        
        # Compute per-class AUROC and AUPRC
        auroc_per_class = roc_auc_score(y_bin, y_proba, average=None, multi_class='ovr')
        auprc_per_class = average_precision_score(y_bin, y_proba, average=None)
        
        print(f"\nAUROC (macro): {auroc_macro:.4f}")
        print(f"AUROC (weighted): {auroc_weighted:.4f}")
        print(f"AUPRC (macro): {auprc_macro:.4f}")
        print(f"AUPRC (weighted): {auprc_weighted:.4f}")
        
        print("\nPer-class AUROC:")
        for i, cls in enumerate(classes):
            print(f"  Class {cls}: {auroc_per_class[i]:.4f}")
        
        print("\nPer-class AUPRC:")
        for i, cls in enumerate(classes):
            print(f"  Class {cls}: {auprc_per_class[i]:.4f}")
        
    except Exception as e:
        print(f"\nWarning: Could not compute AUROC/AUPRC: {e}")
        auroc_macro = auroc_weighted = auprc_macro = auprc_weighted = None
        auroc_per_class = auprc_per_class = None
    
    # Classification Report
    report = classification_report(y, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Save to file if provided
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Evaluation on {dataset_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            
            if auroc_macro is not None:
                f.write(f"\nAUROC (macro): {auroc_macro:.4f}\n")
                f.write(f"AUROC (weighted): {auroc_weighted:.4f}\n")
                f.write(f"AUPRC (macro): {auprc_macro:.4f}\n")
                f.write(f"AUPRC (weighted): {auprc_weighted:.4f}\n")
                
                f.write("\nPer-class AUROC:\n")
                for i, cls in enumerate(classes):
                    f.write(f"  Class {cls}: {auroc_per_class[i]:.4f}\n")
                
                f.write("\nPer-class AUPRC:\n")
                for i, cls in enumerate(classes):
                    f.write(f"  Class {cls}: {auprc_per_class[i]:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write(report + "\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm) + "\n")
    
    return accuracy, y_pred

def show_feature_importance(model, feature_names, top_n=20):
    """Show feature importance for tree-based models, highlighting text vs numeric"""
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*60)
        print(f"Top {top_n} Most Important Features:")
        print("="*60)
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Type': ['Text' if str(f).startswith('tfidf_') else 'Numeric' for f in feature_names]
        }).sort_values(by='Importance', ascending=False).head(top_n)
        
        print(feature_importance_df.to_string(index=False))
        
        # Summary statistics
        text_features = feature_importance_df[feature_importance_df['Type'] == 'Text'].shape[0]
        numeric_features = top_n - text_features
        print(f"\nTop {top_n} features: {text_features} Text, {numeric_features} Numeric")
        
        # Total importance by type
        all_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Type': ['Text' if str(f).startswith('tfidf_') else 'Numeric' for f in feature_names]
        })
        type_importance = all_importance_df.groupby('Type')['Importance'].sum()
        print(f"\nTotal importance by type:")
        for ftype, imp in type_importance.items():
            pct = (imp / importances.sum()) * 100
            print(f"  {ftype}: {imp:.4f} ({pct:.1f}%)")

def save_model(model, filename):
    """Save trained model to disk"""
    models_dir = "ml_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")

def main():
    print("Loading data...")
    train_df, valid_df, test_df = load_data()
    
    # Create per-model results helpers
    os.makedirs("training_results", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_results_file(model_name):
        safe = model_name.lower().replace(" ", "_")
        return os.path.join("training_results", f"{safe}_{timestamp}.txt")

    def init_results_file(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("MODEL TRAINING AND EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
    
    # Prepare data
    X_train, y_train = prepare_features_target(train_df)
    X_valid, y_valid = prepare_features_target(valid_df)
    X_test, y_test = prepare_features_target(test_df)
    
    print(f"\nTotal features: {len(X_train.columns)}")
    tfidf_features = [f for f in X_train.columns if str(f).startswith('tfidf_')]
    print(f"Sample TF-IDF features: {tfidf_features[:5]}")
    
    # Ask user if they want to merge classes
    print("\nMerge acuity classes 3, 4, 5 into a single class?")
    print("(This can improve accuracy due to severe class imbalance)")
    print("1. Yes - Merge to 3 classes (1, 2, 3)")
    print("2. No - Keep original 5 classes")
    merge_choice = input("Enter choice (1 or 2): ").strip()
    
    merge_classes = merge_choice == '1'
    
    if merge_classes:
        # Re-prepare with merged classes
        X_train, y_train = prepare_features_target(train_df, merge_classes=True)
        X_valid, y_valid = prepare_features_target(valid_df, merge_classes=True)
        X_test, y_test = prepare_features_target(test_df, merge_classes=True)
    
    print(f"\nTarget distribution in training set:")
    print(y_train.value_counts().sort_index())
    
    # Ask user for model choice
    print("\nChoose model(s) to train:")
    print("1. Random Forest")
    print("2. Logistic Regression")
    print("3. XGBoost")
    print("4. MLP (Neural Network)")
    print("5. CatBoost")
    print("6. LightGBM")
    print("7. AdaBoost")
    print("8. SVM (Support Vector Machine)")
    print("9. All models (EXCEPT SVM)")
    print("10. All models (INCLUDING SVM)")
    choice = input("Enter choice (1-10): ").strip()
    
    do_train_rf = choice in ['1', '9', '10']
    do_train_lr = choice in ['2', '9', '10']
    do_train_xgb = choice in ['3', '9', '10']
    do_train_mlp = choice in ['4', '9', '10']
    do_train_catboost = choice in ['5', '9', '10']
    do_train_lightgbm = choice in ['6', '9', '10']
    do_train_adaboost = choice in ['7', '9', '10']
    do_train_svm = choice in ['8', '10']
    
    if choice == '9':
        do_train_rf = do_train_lr = do_train_xgb = do_train_mlp = do_train_catboost = do_train_lightgbm = do_train_adaboost = True
        do_train_svm = False
    elif choice == '10':
        do_train_rf = do_train_lr = do_train_xgb = do_train_mlp = do_train_catboost = do_train_lightgbm = do_train_adaboost = do_train_svm = True

    # Global tuning choice for "All" options
    # Per-model individual tuning selection is always used.
    if choice in ['9', '10']:
        print("\nYou have selected all models.")
        print("Hyperparameter tuning will be chosen for each model individually.")

    def select_best_tuned_model(model_name, tuning_options, X_train, y_train, X_valid, y_valid,
                                is_xgb=False, is_catboost=False, is_lightgbm=False, output_file=None):
        """Run all selected tuning techniques and keep the best model by validation accuracy."""
        print("\n" + "="*60)
        print(f"Running all tuning techniques for {model_name}")
        print("="*60)

        best_method = None
        best_model = None
        best_accuracy = -1.0
        tuning_results = []

        for method_name, tuner_fn in tuning_options:
            print(f"\nTrying {method_name} tuning for {model_name}...")
            tuned_model = tuner_fn(X_train, y_train)
            y_pred = tuned_model.predict(X_valid)

            if is_xgb or is_catboost or is_lightgbm:
                y_pred = y_pred + 1

            accuracy = accuracy_score(y_valid, y_pred)
            print(f"Validation accuracy with {method_name}: {accuracy:.4f}")
            tuning_results.append({
                'method': method_name,
                'validation_accuracy': accuracy,
                'params': get_model_params(tuned_model)
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = tuned_model
                best_method = method_name

        print(f"\nSelected best tuning for {model_name}: {best_method} (accuracy={best_accuracy:.4f})")
        attach_training_summary(
            best_model,
            model_name=model_name,
            strategy=f"all tuning techniques -> selected {best_method}",
            best_validation_accuracy=best_accuracy,
            tuning_results=tuning_results
        )
        write_training_summary(output_file, best_model)
        return best_model

    def finalize_model(model, model_name, tuning_method, output_file):
        """Attach and persist summary for single-strategy training runs."""
        strategy_map = {
            None: 'default parameters',
            'random': 'RandomizedSearchCV',
            'grid': 'GridSearchCV',
            'bayesian': 'Bayesian Optimization (Optuna)'
        }
        attach_training_summary(
            model,
            model_name=model_name,
            strategy=strategy_map.get(tuning_method, 'default parameters')
        )
        write_training_summary(output_file, model)
        return model

    # Helper for generic tuning prompt
    def get_tuning_choice(model_name):
        print(f"\nHyperparameter Tuning for {model_name}:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (Massive range)")
        print("3. GridSearchCV (Restricted range for speed)")
        print("4. Bayesian Optimization (Optuna - Wide range)")
        print("5. All tuning techniques (pick best on validation set)")
        tc = input("Enter choice (1-5): ").strip()
        return {'2': 'random', '3': 'grid', '4': 'bayesian', '5': 'all'}.get(tc, None)

    rf_tuning_method = get_tuning_choice("Random Forest") if do_train_rf else None
    lr_tuning_method = get_tuning_choice("Logistic Regression") if do_train_lr else None
    xgb_tuning_method = get_tuning_choice("XGBoost") if do_train_xgb else None
    mlp_tuning_method = get_tuning_choice("MLP") if do_train_mlp else None
    cat_tuning_method = get_tuning_choice("CatBoost") if do_train_catboost else None
    lgb_tuning_method = get_tuning_choice("LightGBM") if do_train_lightgbm else None
    ada_tuning_method = get_tuning_choice("AdaBoost") if do_train_adaboost else None
    svm_tuning_method = get_tuning_choice("SVM") if do_train_svm else None

    rf_results_file = get_results_file("Random Forest") if do_train_rf else None
    lr_results_file = get_results_file("Logistic Regression") if do_train_lr else None
    xgb_results_file = get_results_file("XGBoost") if do_train_xgb else None
    mlp_results_file = get_results_file("MLP") if do_train_mlp else None
    cat_results_file = get_results_file("CatBoost") if do_train_catboost else None
    lgb_results_file = get_results_file("LightGBM") if do_train_lightgbm else None
    ada_results_file = get_results_file("AdaBoost") if do_train_adaboost else None
    svm_results_file = get_results_file("SVM") if do_train_svm else None

    for path in [rf_results_file, lr_results_file, xgb_results_file, mlp_results_file,
                 cat_results_file, lgb_results_file, ada_results_file, svm_results_file]:
        if path:
            init_results_file(path)
    
    rf_model = None
    lr_model = None
    xgb_model = None
    mlp_model = None
    catboost_model = None
    lightgbm_model = None
    adaboost_model = None
    svm_model = None
    
    # Train Random Forest
    if do_train_rf:
        if rf_tuning_method == 'random':
            rf_model = tune_random_forest_random(X_train, y_train, n_iter=100)
        elif rf_tuning_method == 'grid':
            rf_model = tune_random_forest_grid(X_train, y_train)
        elif rf_tuning_method == 'bayesian':
            rf_model = tune_random_forest_bayesian(X_train, y_train, n_trials=50)
        elif rf_tuning_method == 'all':
            rf_model = select_best_tuned_model(
                "Random Forest",
                [
                    ("random", lambda Xt, yt: tune_random_forest_random(Xt, yt, n_iter=100)),
                    ("grid", lambda Xt, yt: tune_random_forest_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_random_forest_bayesian(Xt, yt, n_trials=50))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                output_file=rf_results_file
            )
        else:
            rf_model = train_random_forest(X_train, y_train)

        if rf_tuning_method != 'all':
            rf_model = finalize_model(rf_model, "Random Forest", rf_tuning_method, rf_results_file)
        
        evaluate_model(rf_model, X_valid, y_valid, "Validation Set (Random Forest)", output_file=rf_results_file)
        show_feature_importance(rf_model, X_train.columns)
    
    # Train Logistic Regression
    if do_train_lr:
        if lr_tuning_method == 'random':
            lr_model = tune_logistic_regression_random(X_train, y_train)
        elif lr_tuning_method == 'grid':
            lr_model = tune_logistic_regression_grid(X_train, y_train)
        elif lr_tuning_method == 'bayesian':
            lr_model = tune_logistic_regression_bayesian(X_train, y_train)
        elif lr_tuning_method == 'all':
            lr_model = select_best_tuned_model(
                "Logistic Regression",
                [
                    ("random", lambda Xt, yt: tune_logistic_regression_random(Xt, yt)),
                    ("grid", lambda Xt, yt: tune_logistic_regression_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_logistic_regression_bayesian(Xt, yt))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                output_file=lr_results_file
            )
        else:
            lr_model = train_logistic_regression(X_train, y_train)

        if lr_tuning_method != 'all':
            lr_model = finalize_model(lr_model, "Logistic Regression", lr_tuning_method, lr_results_file)

        evaluate_model(lr_model, X_valid, y_valid, "Validation Set (Logistic Regression)", output_file=lr_results_file)

    # Train XGBoost
    if do_train_xgb:
        if xgb_tuning_method == 'random':
            xgb_model = tune_xgboost_random(X_train, y_train, n_iter=100)
        elif xgb_tuning_method == 'grid':
            xgb_model = tune_xgboost_grid(X_train, y_train)
        elif xgb_tuning_method == 'bayesian':
            xgb_model = tune_xgboost_bayesian(X_train, y_train, n_trials=100)
        elif xgb_tuning_method == 'all':
            xgb_model = select_best_tuned_model(
                "XGBoost",
                [
                    ("random", lambda Xt, yt: tune_xgboost_random(Xt, yt, n_iter=100)),
                    ("grid", lambda Xt, yt: tune_xgboost_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_xgboost_bayesian(Xt, yt, n_trials=100))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                is_xgb=True,
                output_file=xgb_results_file
            )
        else:
            xgb_model = train_xgboost(X_train, y_train)

        if xgb_tuning_method != 'all':
            xgb_model = finalize_model(xgb_model, "XGBoost", xgb_tuning_method, xgb_results_file)
        
        evaluate_model(xgb_model, X_valid, y_valid, "Validation Set (XGBoost)", is_xgb=True, output_file=xgb_results_file)
        show_feature_importance(xgb_model, X_train.columns)
    
    # Train MLP
    if do_train_mlp:
        if mlp_tuning_method == 'random':
            mlp_model = tune_mlp_random(X_train, y_train, n_iter=100)
        elif mlp_tuning_method == 'grid':
            mlp_model = tune_mlp_grid(X_train, y_train)
        elif mlp_tuning_method == 'bayesian':
            mlp_model = tune_mlp_bayesian(X_train, y_train, n_trials=50)
        elif mlp_tuning_method == 'all':
            mlp_model = select_best_tuned_model(
                "MLP",
                [
                    ("random", lambda Xt, yt: tune_mlp_random(Xt, yt, n_iter=100)),
                    ("grid", lambda Xt, yt: tune_mlp_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_mlp_bayesian(Xt, yt, n_trials=50))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                output_file=mlp_results_file
            )
        else:
            mlp_model = train_mlp(X_train, y_train)

        if mlp_tuning_method != 'all':
            mlp_model = finalize_model(mlp_model, "MLP", mlp_tuning_method, mlp_results_file)
        
        evaluate_model(mlp_model, X_valid, y_valid, "Validation Set (MLP)", output_file=mlp_results_file)
        
    # Train CatBoost
    if do_train_catboost:
        if cat_tuning_method == 'random':
            catboost_model = tune_catboost_random(X_train, y_train)
        elif cat_tuning_method == 'grid':
            catboost_model = tune_catboost_grid(X_train, y_train)
        elif cat_tuning_method == 'bayesian':
            catboost_model = tune_catboost_bayesian(X_train, y_train)
        elif cat_tuning_method == 'all':
            catboost_model = select_best_tuned_model(
                "CatBoost",
                [
                    ("random", lambda Xt, yt: tune_catboost_random(Xt, yt)),
                    ("grid", lambda Xt, yt: tune_catboost_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_catboost_bayesian(Xt, yt))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                is_catboost=True,
                output_file=cat_results_file
            )
        else:
            catboost_model = train_catboost(X_train, y_train)

        if cat_tuning_method != 'all':
            catboost_model = finalize_model(catboost_model, "CatBoost", cat_tuning_method, cat_results_file)

        evaluate_model(catboost_model, X_valid, y_valid, "Validation Set (CatBoost)", is_catboost=True, output_file=cat_results_file)
        show_feature_importance(catboost_model, X_train.columns)
    
    # Train LightGBM
    if do_train_lightgbm:
        if lgb_tuning_method == 'random':
            lightgbm_model = tune_lightgbm_random(X_train, y_train, n_iter=100)
        elif lgb_tuning_method == 'grid':
            lightgbm_model = tune_lightgbm_grid(X_train, y_train)
        elif lgb_tuning_method == 'bayesian':
            lightgbm_model = tune_lightgbm_bayesian(X_train, y_train, n_trials=100)
        elif lgb_tuning_method == 'all':
            lightgbm_model = select_best_tuned_model(
                "LightGBM",
                [
                    ("random", lambda Xt, yt: tune_lightgbm_random(Xt, yt, n_iter=100)),
                    ("grid", lambda Xt, yt: tune_lightgbm_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_lightgbm_bayesian(Xt, yt, n_trials=100))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                is_lightgbm=True,
                output_file=lgb_results_file
            )
        else:
            lightgbm_model = train_lightgbm(X_train, y_train)

        if lgb_tuning_method != 'all':
            lightgbm_model = finalize_model(lightgbm_model, "LightGBM", lgb_tuning_method, lgb_results_file)
        
        evaluate_model(lightgbm_model, X_valid, y_valid, "Validation Set (LightGBM)", is_lightgbm=True, output_file=lgb_results_file)
        show_feature_importance(lightgbm_model, X_train.columns)
    
    # Train AdaBoost
    if do_train_adaboost:
        if ada_tuning_method == 'random':
            adaboost_model = tune_adaboost_random(X_train, y_train)
        elif ada_tuning_method == 'grid':
            adaboost_model = tune_adaboost_grid(X_train, y_train)
        elif ada_tuning_method == 'bayesian':
            adaboost_model = tune_adaboost_bayesian(X_train, y_train)
        elif ada_tuning_method == 'all':
            adaboost_model = select_best_tuned_model(
                "AdaBoost",
                [
                    ("random", lambda Xt, yt: tune_adaboost_random(Xt, yt)),
                    ("grid", lambda Xt, yt: tune_adaboost_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_adaboost_bayesian(Xt, yt))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                output_file=ada_results_file
            )
        else:
            adaboost_model = train_adaboost(X_train, y_train)

        if ada_tuning_method != 'all':
            adaboost_model = finalize_model(adaboost_model, "AdaBoost", ada_tuning_method, ada_results_file)

        evaluate_model(adaboost_model, X_valid, y_valid, "Validation Set (AdaBoost)", output_file=ada_results_file)

    # Train SVM
    if do_train_svm:
        if svm_tuning_method == 'random':
            svm_model = tune_svm_random(X_train, y_train)
        elif svm_tuning_method == 'grid':
            svm_model = tune_svm_grid(X_train, y_train)
        elif svm_tuning_method == 'bayesian':
            svm_model = tune_svm_bayesian(X_train, y_train)
        elif svm_tuning_method == 'all':
            svm_model = select_best_tuned_model(
                "SVM",
                [
                    ("random", lambda Xt, yt: tune_svm_random(Xt, yt)),
                    ("grid", lambda Xt, yt: tune_svm_grid(Xt, yt)),
                    ("bayesian", lambda Xt, yt: tune_svm_bayesian(Xt, yt))
                ],
                X_train,
                y_train,
                X_valid,
                y_valid,
                output_file=svm_results_file
            )
        else:
            svm_model = train_svm(X_train, y_train)

        if svm_tuning_method != 'all':
            svm_model = finalize_model(svm_model, "SVM", svm_tuning_method, svm_results_file)

        evaluate_model(svm_model, X_valid, y_valid, "Validation Set (SVM)", output_file=svm_results_file)
    
    # Final evaluation on test set
    print("\n" + "#"*60)
    print("FINAL EVALUATION ON TEST SET")
    print("#"*60)
    
    if rf_model:
        print("\nRandom Forest on Test Set:")
        evaluate_model(rf_model, X_test, y_test, "Test Set (Random Forest)", output_file=rf_results_file)
    
    if lr_model:
        print("\nLogistic Regression on Test Set:")
        evaluate_model(lr_model, X_test, y_test, "Test Set (Logistic Regression)", output_file=lr_results_file)
    
    if xgb_model:
        print("\nXGBoost on Test Set:")
        evaluate_model(xgb_model, X_test, y_test, "Test Set (XGBoost)", is_xgb=True, output_file=xgb_results_file)
    
    if mlp_model:
        print("\nMLP on Test Set:")
        evaluate_model(mlp_model, X_test, y_test, "Test Set (MLP)", output_file=mlp_results_file)
    
    if catboost_model:
        print("\nCatBoost on Test Set:")
        evaluate_model(catboost_model, X_test, y_test, "Test Set (CatBoost)", is_catboost=True, output_file=cat_results_file)
    
    if lightgbm_model:
        print("\nLightGBM on Test Set:")
        evaluate_model(lightgbm_model, X_test, y_test, "Test Set (LightGBM)", is_lightgbm=True, output_file=lgb_results_file)
    
    if adaboost_model:
        print("\nAdaBoost on Test Set:")
        evaluate_model(adaboost_model, X_test, y_test, "Test Set (AdaBoost)", output_file=ada_results_file)

    if svm_model:
        print("\nSVM on Test Set:")
        evaluate_model(svm_model, X_test, y_test, "Test Set (SVM)", output_file=svm_results_file)
    
    print(f"\n{'='*60}")
    print("Results saved to:")
    for label, path in [
        ("Random Forest", rf_results_file),
        ("Logistic Regression", lr_results_file),
        ("XGBoost", xgb_results_file),
        ("MLP", mlp_results_file),
        ("CatBoost", cat_results_file),
        ("LightGBM", lgb_results_file),
        ("AdaBoost", ada_results_file),
        ("SVM", svm_results_file),
    ]:
        if path:
            print(f"- {label}: {path}")
    print("="*60)

if __name__ == "__main__":
    main()
