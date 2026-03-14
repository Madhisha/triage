import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform, loguniform

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight

def tune_lightgbm_random(X_train, y_train, n_iter=100):
    """Tune LightGBM using RandomizedSearchCV with distribution sampling."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (RandomizedSearchCV - Massive)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    param_distributions = {
        'n_estimators': randint(100, 5001),
        'max_depth': [-1] + np.arange(5, 51, 5).tolist(),
        'learning_rate': loguniform(1e-4, 0.5),
        'num_leaves': randint(20, 501),
        'min_child_samples': randint(5, 101),
        'subsample': uniform(0.2, 0.8),
        'colsample_bytree': uniform(0.2, 0.8),
        'reg_alpha': loguniform(1e-8, 10.0),
        'reg_lambda': loguniform(1e-8, 10.0),
        'boosting_type': ['gbdt', 'dart']
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
    """Tune LightGBM using GridSearchCV with expanded grid."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (GridSearchCV - Massive)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    param_grid = {
        'n_estimators': [200, 500, 1000, 2000, 3000, 5000],
        'max_depth': [-1, 8, 12, 16, 24, 32, 50],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 63, 127, 255, 511],
        'min_child_samples': [5, 10, 20, 30, 50, 100],
        'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
        'boosting_type': ['gbdt', 'dart']
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


def tune_lightgbm_bayesian(X_train, y_train, n_trials=100):
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
            'max_depth': trial.suggest_categorical('max_depth', [-1] + list(range(5, 51, 5))),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
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





def train_lightgbm(X_train, y_train):
    """Train LightGBM Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training LightGBM Classifier (default parameters)...")
    print("="*60)
    
    # Convert labels to 0-indexed for LightGBM
    y_train_lgb = y_train - 1
    
    lgb_model = LGBMClassifier()
    lgb_model.fit(X_train, y_train_lgb)
    print("LightGBM training completed.")
    return lgb_model

