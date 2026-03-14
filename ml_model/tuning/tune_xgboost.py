import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform, loguniform

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

def tune_xgboost_random(X_train, y_train, n_iter=100):
    """Tune XGBoost using RandomizedSearchCV with distribution sampling."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (RandomizedSearchCV - Massive)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_distributions = {
        'n_estimators': randint(100, 5001),
        'max_depth': randint(2, 22),
        'learning_rate': loguniform(1e-4, 0.5),
        'subsample': uniform(0.2, 0.8),
        'colsample_bytree': uniform(0.2, 0.8),
        'colsample_bylevel': uniform(0.2, 0.8),
        'min_child_weight': randint(1, 31),
        'gamma': uniform(0.0, 10.0),
        'reg_alpha': loguniform(1e-8, 10.0),
        'reg_lambda': loguniform(1e-8, 10.0),
        'scale_pos_weight': [1, 2, 5, 10],
        'booster': ['gbtree', 'dart']
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
    """Tune XGBoost using GridSearchCV with expanded grid."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (GridSearchCV - Massive)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_grid = {
        'n_estimators': [200, 500, 1000, 2000, 3000, 5000],
        'max_depth': [3, 5, 7, 10, 14, 18, 22],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10, 15],
        'gamma': [0, 0.1, 0.2, 0.5, 1.0]
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


def tune_xgboost_bayesian(X_train, y_train, n_trials=100):
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'gamma': trial.suggest_float('gamma', 0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
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




def train_xgboost(X_train, y_train):
    """Train XGBoost Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training XGBoost Classifier (default parameters)...")
    print("="*60)
    
    # Convert labels to 0-indexed for XGBoost (1,2,3,4,5 -> 0,1,2,3,4)
    y_train_xgb = y_train - 1
    
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train_xgb)
    print("XGBoost training completed.")
    return xgb_model

