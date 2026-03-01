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
import joblib
import os
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

def load_data():
    """Load the preprocessed train, validation, and test datasets with chief complaint features"""
    train_df = pd.read_csv("ml_processed_data/ml_processed_train.csv")
    valid_df = pd.read_csv("ml_processed_data/ml_processed_valid.csv")
    test_df = pd.read_csv("ml_processed_data/ml_processed_test.csv")

    # train_df = pd.read_csv("ml_processed_data/balanced/ml_processed_train.csv")
    # valid_df = pd.read_csv("ml_processed_data/balanced/ml_processed_valid.csv")
    # test_df = pd.read_csv("ml_processed_data/balanced/ml_processed_test.csv")
    
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

def tune_random_forest_random(X_train, y_train, n_iter=20):
    """Tune Random Forest using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (RandomizedSearchCV)")
    print("="*60)
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 30, 50],
    }
    
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_random_forest_grid(X_train, y_train):
    """Tune Random Forest using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (GridSearchCV)")
    print("="*60)
    
    param_grid = {
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 30],
    }
    
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_random_forest_bayesian(X_train, y_train, n_trials=30):
    """Tune Random Forest using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_random_forest_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (Bayesian - Optuna)")
    print("="*60)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 30, 50]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf = RandomForestClassifier(**params)
        
        # Simple cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model

# ==================== XGBoost Tuning ====================

def tune_xgboost_random(X_train, y_train, n_iter=20):
    """Tune XGBoost using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (RandomizedSearchCV)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [3, 5, 7, 10, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
    }
    
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_xgboost_grid(X_train, y_train):
    """Tune XGBoost using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (GridSearchCV)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_grid = {
        'n_estimators': [200, 300, 500, 800, 1000],
        'max_depth': [7, 10, 12],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.6, 0.8],
        'min_child_weight': [3, 5],
    }
    
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_xgboost_bayesian(X_train, y_train, n_trials=30):
    """Tune XGBoost using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_xgboost_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (Bayesian - Optuna)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        
        xgb = XGBClassifier(**params)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(xgb, X_train, y_train_xgb, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'mlogloss'
    
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    
    return best_model

# ==================== MLP Tuning ====================

def tune_mlp_random(X_train, y_train, n_iter=20):
    """Tune MLP using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (RandomizedSearchCV)")
    print("="*60)
    
    param_distributions = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64), (512, 256, 128)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'batch_size': ['auto', 32, 64],
    }
    
    mlp_base = MLPClassifier(
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    
    random_search = RandomizedSearchCV(
        mlp_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_mlp_grid(X_train, y_train):
    """Tune MLP using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (GridSearchCV)")
    print("="*60)
    
    param_grid = {
        'hidden_layer_sizes': [(256, 128, 64), (512, 256, 128)],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.001],
    }
    
    mlp_base = MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    
    grid_search = GridSearchCV(
        mlp_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_mlp_bayesian(X_train, y_train, n_trials=30):
    """Tune MLP using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_mlp_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (Bayesian - Optuna)")
    print("="*60)
    
    def objective(trial):
        # Suggest network architecture
        n_layers = trial.suggest_int('n_layers', 2, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_categorical(f'n_units_l{i}', [64, 128, 256, 512]))
        
        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 'auto']),
            'solver': 'adam',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
            'tol': 1e-4
        }
        
        mlp = MLPClassifier(**params)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(mlp, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    # Reconstruct best architecture
    best_params = study.best_params
    n_layers = best_params['n_layers']
    layers = tuple([best_params[f'n_units_l{i}'] for i in range(n_layers)])
    
    best_model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        batch_size=best_params['batch_size'],
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    best_model.fit(X_train, y_train)
    
    return best_model

# ==================== LightGBM Tuning ====================

def tune_lightgbm_random(X_train, y_train, n_iter=20):
    """Tune LightGBM using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (RandomizedSearchCV)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [5, 10, 15, 20, -1],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [20, 31, 50, 70],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }
    
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_lightgbm_grid(X_train, y_train):
    """Tune LightGBM using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (GridSearchCV)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    param_grid = {
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [10, 15],
        'learning_rate': [0.03, 0.05],
        'num_leaves': [31, 50],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    grid_search = GridSearchCV(
        lgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_lightgbm_bayesian(X_train, y_train, n_trials=30):
    """Tune LightGBM using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_lightgbm_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (Bayesian - Optuna)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        lgb = LGBMClassifier(**params)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(lgb, X_train, y_train_lgb, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    
    return best_model

def train_random_forest(X_train, y_train, max_features='log2', n_estimators=1000):
    """Train Random Forest Classifier (optimized for text + numeric features)"""
    print("\n" + "="*60)
    print("Training Random Forest Classifier (with Chief Complaint)...")
    print("="*60)
    print(f"Note: Using max_features='{max_features}', n_estimators={n_estimators}")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=max_features,  # Tunable parameter
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return rf_model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression (works well with TF-IDF features)"""
    print("\n" + "="*60)
    print("Training Logistic Regression (with Chief Complaint)...")
    print("="*60)
    print("Note: LR is effective for text classification tasks.")
    
    lr_model = LogisticRegression(
        max_iter=2000,  # Increased for convergence with more features
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        C=1.0,  # Regularization strength
        random_state=42,
        verbose=1
    )
    
    lr_model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return lr_model

def train_xgboost(X_train, y_train, n_estimators=1000):
    """Train XGBoost Classifier (handles sparse features well)"""
    print("\n" + "="*60)
    print("Training XGBoost Classifier (with Chief Complaint)...")
    print("="*60)
    print(f"Note: Using n_estimators={n_estimators}")
    
    # Convert labels to 0-indexed for XGBoost (1,2,3,4,5 -> 0,1,2,3,4)
    y_train_xgb = y_train - 1
    
    # Compute class weights for XGBoost
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,  # Increased for complex patterns
        max_depth=12,  # Moderate depth for text features
        learning_rate=0.05,  # Lower for better generalization
        subsample=0.8,
        colsample_bytree=0.6,  # Reduced to handle many features
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print("XGBoost training completed.")
    return xgb_model

def train_mlp(X_train, y_train):
    """Train Multi-Layer Perceptron (good for text + numeric features)"""
    print("\n" + "="*60)
    print("Training MLP Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: Neural networks can learn complex text patterns.")
    print("Using early stopping to prevent overfitting.")
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # Larger layers for more features
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.0001,
        max_iter=500,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    
    mlp_model.fit(X_train, y_train)
    print("MLP training completed.")
    return mlp_model

def train_catboost(X_train, y_train):
    """Train CatBoost Classifier (handles categorical and text features natively)"""
    print("\n" + "="*60)
    print("Training CatBoost Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: CatBoost is robust and handles imbalanced data well.")
    
    # Convert labels to 0-indexed for CatBoost
    y_train_cat = y_train - 1
    
    # Compute class weights
    classes = np.unique(y_train_cat)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cat)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        class_weights=class_weight_dict,
        random_seed=42,
        verbose=100,
        loss_function='MultiClass'
    )
    
    cat_model.fit(X_train, y_train_cat)
    print("CatBoost training completed.")
    return cat_model

def train_lightgbm(X_train, y_train):
    """Train LightGBM Classifier (fast and efficient for large datasets)"""
    print("\n" + "="*60)
    print("Training LightGBM Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: LightGBM is very fast and memory efficient.")
    
    # Convert labels to 0-indexed for LightGBM
    y_train_lgb = y_train - 1
    
    # Compute class weights
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        num_leaves=20,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print("LightGBM training completed.")
    return lgb_model

def train_adaboost(X_train, y_train):
    """Train AdaBoost Classifier (adaptive boosting)"""
    print("\n" + "="*60)
    print("Training AdaBoost Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: AdaBoost focuses on misclassified samples.")
    
    ada_model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    
    ada_model.fit(X_train, y_train)
    print("AdaBoost training completed.")
    return ada_model

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
    
    # Create results output file
    results_file = "training_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
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
    print("\nChoose model to train:")
    print("1. Random Forest")
    print("2. Logistic Regression")
    print("3. XGBoost")
    print("4. MLP (Neural Network)")
    print("5. CatBoost")
    print("6. LightGBM")
    print("7. AdaBoost")
    print("8. All models")
    choice = input("Enter choice (1-8): ").strip()
    
    do_train_rf = choice in ['1', '8']
    do_train_lr = choice in ['2', '8']
    do_train_xgb = choice in ['3', '8']
    do_train_mlp = choice in ['4', '8']
    do_train_catboost = choice in ['5', '8']
    do_train_lightgbm = choice in ['6', '8']
    do_train_adaboost = choice in ['7', '8']
    
    if not (do_train_rf or do_train_lr or do_train_xgb or do_train_mlp or do_train_catboost or do_train_lightgbm or do_train_adaboost):
        print("Invalid choice. Defaulting to Random Forest.")
        do_train_rf = True
    
    # Ask user for hyperparameter tuning methods for supported models
    rf_tuning_method = None
    xgb_tuning_method = None
    mlp_tuning_method = None
    lgb_tuning_method = None
    
    if do_train_rf:
        print("\nHyperparameter Tuning for Random Forest:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (fast, explores random combinations)")
        print("3. GridSearchCV (slower, exhaustive search)")
        print("4. Bayesian Optimization (Optuna - smart search)")
        tuning_choice = input("Enter choice (1-4): ").strip()
        
        if tuning_choice == '2':
            rf_tuning_method = 'random'
        elif tuning_choice == '3':
            rf_tuning_method = 'grid'
        elif tuning_choice == '4':
            rf_tuning_method = 'bayesian'
    
    if do_train_xgb:
        print("\nHyperparameter Tuning for XGBoost:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (fast, explores random combinations)")
        print("3. GridSearchCV (slower, exhaustive search)")
        print("4. Bayesian Optimization (Optuna - smart search)")
        tuning_choice = input("Enter choice (1-4): ").strip()
        
        if tuning_choice == '2':
            xgb_tuning_method = 'random'
        elif tuning_choice == '3':
            xgb_tuning_method = 'grid'
        elif tuning_choice == '4':
            xgb_tuning_method = 'bayesian'
    
    if do_train_mlp:
        print("\nHyperparameter Tuning for MLP:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (fast, explores random combinations)")
        print("3. GridSearchCV (slower, exhaustive search)")
        print("4. Bayesian Optimization (Optuna - smart search)")
        tuning_choice = input("Enter choice (1-4): ").strip()
        
        if tuning_choice == '2':
            mlp_tuning_method = 'random'
        elif tuning_choice == '3':
            mlp_tuning_method = 'grid'
        elif tuning_choice == '4':
            mlp_tuning_method = 'bayesian'
    
    if do_train_lightgbm:
        print("\nHyperparameter Tuning for LightGBM:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (fast, explores random combinations)")
        print("3. GridSearchCV (slower, exhaustive search)")
        print("4. Bayesian Optimization (Optuna - smart search)")
        tuning_choice = input("Enter choice (1-4): ").strip()
        
        if tuning_choice == '2':
            lgb_tuning_method = 'random'
        elif tuning_choice == '3':
            lgb_tuning_method = 'grid'
        elif tuning_choice == '4':
            lgb_tuning_method = 'bayesian'
    
    rf_model = None
    lr_model = None
    xgb_model = None
    mlp_model = None
    catboost_model = None
    lightgbm_model = None
    adaboost_model = None
    
    # Train Random Forest
    if do_train_rf:
        if rf_tuning_method == 'random':
            rf_model = tune_random_forest_random(X_train, y_train, n_iter=20)
        elif rf_tuning_method == 'grid':
            rf_model = tune_random_forest_grid(X_train, y_train)
        elif rf_tuning_method == 'bayesian':
            rf_model = tune_random_forest_bayesian(X_train, y_train, n_trials=30)
        else:
            rf_model = train_random_forest(X_train, y_train)
        
        evaluate_model(rf_model, X_valid, y_valid, "Validation Set (Random Forest)", output_file=results_file)
        show_feature_importance(rf_model, X_train.columns)
        # save_model(rf_model, "ml_random_forest_model.pkl")
    
    # Train Logistic Regression
    if do_train_lr:
        lr_model = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_valid, y_valid, "Validation Set (Logistic Regression)", output_file=results_file)
        # save_model(lr_model, "ml_logistic_regression_model.pkl")

    # Train XGBoost
    if do_train_xgb:
        if xgb_tuning_method == 'random':
            xgb_model = tune_xgboost_random(X_train, y_train, n_iter=20)
        elif xgb_tuning_method == 'grid':
            xgb_model = tune_xgboost_grid(X_train, y_train)
        elif xgb_tuning_method == 'bayesian':
            xgb_model = tune_xgboost_bayesian(X_train, y_train, n_trials=30)
        else:
            xgb_model = train_xgboost(X_train, y_train)
        
        evaluate_model(xgb_model, X_valid, y_valid, "Validation Set (XGBoost)", is_xgb=True, output_file=results_file)
        show_feature_importance(xgb_model, X_train.columns)
        # save_model(xgb_model, "ml_xgboost_model.pkl")
    
    # Train MLP
    if do_train_mlp:
        if mlp_tuning_method == 'random':
            mlp_model = tune_mlp_random(X_train, y_train, n_iter=20)
        elif mlp_tuning_method == 'grid':
            mlp_model = tune_mlp_grid(X_train, y_train)
        elif mlp_tuning_method == 'bayesian':
            mlp_model = tune_mlp_bayesian(X_train, y_train, n_trials=30)
        else:
            mlp_model = train_mlp(X_train, y_train)
        
        evaluate_model(mlp_model, X_valid, y_valid, "Validation Set (MLP)", output_file=results_file)
        
    # Train CatBoost
    if do_train_catboost:
        catboost_model = train_catboost(X_train, y_train)
        evaluate_model(catboost_model, X_valid, y_valid, "Validation Set (CatBoost)", is_catboost=True, output_file=results_file)
        show_feature_importance(catboost_model, X_train.columns)
        # save_model(catboost_model, "ml_catboost_model.pkl")
    
    # Train LightGBM
    if do_train_lightgbm:
        if lgb_tuning_method == 'random':
            lightgbm_model = tune_lightgbm_random(X_train, y_train, n_iter=20)
        elif lgb_tuning_method == 'grid':
            lightgbm_model = tune_lightgbm_grid(X_train, y_train)
        elif lgb_tuning_method == 'bayesian':
            lightgbm_model = tune_lightgbm_bayesian(X_train, y_train, n_trials=30)
        else:
            lightgbm_model = train_lightgbm(X_train, y_train)
        
        evaluate_model(lightgbm_model, X_valid, y_valid, "Validation Set (LightGBM)", is_lightgbm=True, output_file=results_file)
        show_feature_importance(lightgbm_model, X_train.columns)
        # save_model(lightgbm_model, "ml_lightgbm_model.pkl")
    
    # Train AdaBoost
    if do_train_adaboost:
        adaboost_model = train_adaboost(X_train, y_train)
        evaluate_model(adaboost_model, X_valid, y_valid, "Validation Set (AdaBoost)", output_file=results_file)
        # save_model(adaboost_model, "ml_adaboost_model.pkl")
    
    # Final evaluation on test set
    print("\n" + "#"*60)
    print("FINAL EVALUATION ON TEST SET")
    print("#"*60)
    
    if do_train_rf:
        print("\nRandom Forest on Test Set:")
        evaluate_model(rf_model, X_test, y_test, "Test Set (Random Forest)", output_file=results_file)
    
    if do_train_lr:
        print("\nLogistic Regression on Test Set:")
        evaluate_model(lr_model, X_test, y_test, "Test Set (Logistic Regression)", output_file=results_file)
    
    if do_train_xgb:
        print("\nXGBoost on Test Set:")
        evaluate_model(xgb_model, X_test, y_test, "Test Set (XGBoost)", is_xgb=True, output_file=results_file)
    
    if do_train_mlp:
        print("\nMLP on Test Set:")
        evaluate_model(mlp_model, X_test, y_test, "Test Set (MLP)", output_file=results_file)
    
    if do_train_catboost:
        print("\nCatBoost on Test Set:")
        evaluate_model(catboost_model, X_test, y_test, "Test Set (CatBoost)", is_catboost=True, output_file=results_file)
    
    if do_train_lightgbm:
        print("\nLightGBM on Test Set:")
        evaluate_model(lightgbm_model, X_test, y_test, "Test Set (LightGBM)", is_lightgbm=True, output_file=results_file)
    
    if do_train_adaboost:
        print("\nAdaBoost on Test Set:")
        evaluate_model(adaboost_model, X_test, y_test, "Test Set (AdaBoost)", output_file=results_file)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print("="*60)

if __name__ == "__main__":
    main()
