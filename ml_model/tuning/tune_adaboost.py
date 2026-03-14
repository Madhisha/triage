from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, loguniform

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.ensemble import AdaBoostClassifier

def tune_adaboost_random(X_train, y_train, n_iter=50):
    """Tune AdaBoost using RandomizedSearchCV with distribution sampling."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: AdaBoost (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = {
        'n_estimators': randint(10, 5001),
        'learning_rate': loguniform(1e-5, 2.0),
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [42]
    }
    
    ada_base = AdaBoostClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        ada_base,
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


def tune_adaboost_grid(X_train, y_train):
    """Tune AdaBoost using GridSearchCV with expanded grid."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: AdaBoost (GridSearchCV - Massive)")
    print("="*60)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000, 2000, 3000, 5000],
        'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    ada_base = AdaBoostClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        ada_base,
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


def tune_adaboost_bayesian(X_train, y_train, n_trials=100):
    """Tune AdaBoost using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Falling back to RandomizedSearchCV...")
        return tune_adaboost_random(X_train, y_train)
        
    print("\n" + "="*60)
    print("Hyperparameter Tuning: AdaBoost (Bayesian - Optuna)")
    print("="*60)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 2.0, log=True),
            'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
            'random_state': 42
        }
        
        ada = AdaBoostClassifier(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(ada, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_model = AdaBoostClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    return best_model




def train_adaboost(X_train, y_train):
    """Train AdaBoost Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training AdaBoost Classifier...")
    print("="*60)
    ada_model = AdaBoostClassifier()
    
    ada_model.fit(X_train, y_train)
    print("AdaBoost training completed.")
    return ada_model

