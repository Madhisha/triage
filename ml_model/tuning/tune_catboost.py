import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

def tune_catboost_random(X_train, y_train, n_iter=50):
    """Tune CatBoost using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: CatBoost (RandomizedSearchCV - Massive)")
    print("="*60)
    
    y_train_cat = y_train - 1
    classes = np.unique(y_train_cat)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cat)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    param_distributions = {
        'iterations': np.arange(100, 5001, 100).tolist(),         # 50
        'depth': np.arange(2, 13, 1).tolist(),                    # 11
        'learning_rate': np.logspace(-4, -0.3, 30).tolist(),      # 30
        'l2_leaf_reg': np.logspace(0, 2, 20).tolist(),            # 20
        'border_count': [32, 64, 128, 255],                       # 4
        'thread_count': [-1],
        'random_seed': [42],
        'subsample': np.arange(0.2, 1.05, 0.2).tolist(),          # 5
        'colsample_bylevel': np.arange(0.2, 1.05, 0.2).tolist(),  # 5
        'bagging_temperature': [0, 0.5, 1, 2, 5],                 # 5
        'random_strength': [1, 5, 10, 20, 50]                     # 5
    }
    # 50 * 11 * 30 * 20 * 4 * 5 * 5 * 5 * 5 = 1,650,000,000 (1.65 Billion combinations)
    
    cat_base = CatBoostClassifier(
        loss_function='MultiClass',
        verbose=False,
        class_weights=class_weight_dict
    )
    
    random_search = RandomizedSearchCV(
        cat_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train_cat)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def tune_catboost_grid(X_train, y_train):
    """Tune CatBoost using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: CatBoost (GridSearchCV - Massive)")
    print("="*60)
    
    y_train_cat = y_train - 1
    classes = np.unique(y_train_cat)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cat)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    param_grid = {
        'iterations': [500, 1000, 2000],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 9],
        'border_count': [32, 64, 128]
    }
    
    cat_base = CatBoostClassifier(
        loss_function='MultiClass',
        verbose=False,
        class_weights=class_weight_dict,
        random_seed=42
    )
    
    grid_search = GridSearchCV(
        cat_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train_cat)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def tune_catboost_bayesian(X_train, y_train, n_trials=50):
    """Tune CatBoost using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Falling back to RandomizedSearchCV...")
        return tune_catboost_random(X_train, y_train)
        
    print("\n" + "="*60)
    print("Hyperparameter Tuning: CatBoost (Bayesian - Optuna)")
    print("="*60)
    
    y_train_cat = y_train - 1
    classes = np.unique(y_train_cat)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cat)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 5000),
            'depth': trial.suggest_int('depth', 2, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 100, log=True),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255]),
            'loss_function': 'MultiClass',
            'class_weights': class_weight_dict,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        
        cat = CatBoostClassifier(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(cat, X_train, y_train_cat, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['loss_function'] = 'MultiClass'
    best_params['class_weights'] = class_weight_dict
    best_params['random_seed'] = 42
    best_params['verbose'] = 100
    
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train_cat)
    return best_model





def train_catboost(X_train, y_train):
    """Train CatBoost Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training CatBoost Classifier (default parameters)...")
    print("="*60)
    
    y_train_cat = y_train - 1
    cat_model = CatBoostClassifier()
    
    cat_model.fit(X_train, y_train_cat)
    print("CatBoost training completed.")
    return cat_model

