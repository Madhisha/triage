import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier

def tune_random_forest_random(X_train, y_train, n_iter=100):
    """Tune Random Forest using RandomizedSearchCV near the int32 combination limit."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (RandomizedSearchCV - ~1.43B combos)")
    print("="*60)
    # Explicit lists sized to stay below 2,147,483,647 total combinations (~1.43B here).
    param_distributions = {
        'n_estimators': np.arange(100, 3600, 50).tolist(),       # 70
        'max_depth': [None] + list(range(2, 42, 2)),             # 21
        'min_samples_split': np.arange(2, 32, 2).tolist(),       # 15
        'min_samples_leaf': np.arange(1, 30, 2).tolist(),        # 15
        'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, None], # 9
        'bootstrap': [True, False],                              # 2
        'criterion': ['gini', 'entropy', 'log_loss'],            # 3
        'max_samples': [None, 0.4, 0.6, 0.8, 0.9],               # 5
        'class_weight': ['balanced', 'balanced_subsample'],      # 2
        'ccp_alpha': np.logspace(-5, -2, 8).tolist()             # 8
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
    """Tune Random Forest using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (GridSearchCV - Massive)")
    print("="*60)
    
    # Note: Using a slightly reduced grid for GridSearch to avoid infinite runtime
    param_grid = {
        'n_estimators': [200, 500, 1000, 2000, 3000],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
        'criterion': ['gini', 'entropy']
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'max_depth': trial.suggest_categorical('max_depth', [None] + list(range(5, 51, 5))),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
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




def train_random_forest(X_train, y_train):
    """Train Random Forest Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training Random Forest Classifier (default parameters)...")
    print("="*60)
    rf_model = RandomForestClassifier()
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return rf_model

