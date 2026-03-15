import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight

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


def train_lightgbm(X_train, y_train):
    """Train LightGBM Classifier (fast and efficient for large datasets)"""
    print("\n" + "="*60)
    print("Training LightGBM Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: LightGBM is very fast and memory efficient.")

    y_train_lgb = y_train - 1

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

