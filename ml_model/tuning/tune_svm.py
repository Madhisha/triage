import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.svm import SVC

def tune_svm_random(X_train, y_train, n_iter=30):
    """Tune SVM using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: SVM (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = {
        'C': np.logspace(-5, 2, 60).tolist(),                     # 60
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],           # 4
        'gamma': ['scale', 'auto'] + np.logspace(-5, 1, 60).tolist(), # 62
        'degree': np.arange(1, 11, 1).tolist(),                   # 10
        'coef0': np.linspace(0, 50, 60).tolist(),                 # 60
        'shrinking': [True, False],                               # 2
        'class_weight': ['balanced', None],                       # 2
        'tol': np.logspace(-5, -2, 10).tolist(),                  # 10
        'probability': [True]
    }
    # 60 * 4 * 62 * 10 * 60 * 2 * 2 * 10 = 357,120,000 (357 Million combinations)
    
    # Bump C and coef0 up to reach closer to 2B limit
    param_distributions['C'] = np.logspace(-5, 3, 100).tolist()           # 100
    param_distributions['coef0'] = np.linspace(0, 50, 100).tolist()       # 100
    # 100 * 4 * 62 * 10 * 100 * 2 * 2 * 10 = 992,000,000 (~1 Billion)
    
    param_distributions['degree'] = np.arange(1, 21, 1).tolist()          # 20
    # 100 * 4 * 62 * 20 * 100 * 2 * 2 * 10 = 1,984,000,000 (~1.98 Billion combinations)
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    
    random_search = RandomizedSearchCV(
        svm_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def tune_svm_grid(X_train, y_train):
    """Tune SVM using GridSearchCV with moderate grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: SVM (GridSearchCV - Moderate)")
    print("="*60)
    
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'probability': [True]
    }
    
    svm_base = SVC(class_weight='balanced', random_state=42)
    
    grid_search = GridSearchCV(
        svm_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def tune_svm_bayesian(X_train, y_train, n_trials=30):
    """Tune SVM using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Falling back to RandomizedSearchCV...")
        return tune_svm_random(X_train, y_train)
        
    print("\n" + "="*60)
    print("Hyperparameter Tuning: SVM (Bayesian - Optuna)")
    print("="*60)
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-5, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma_type', ['scale', 'auto', 'custom']),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        if params['gamma'] == 'custom':
            params['gamma'] = trial.suggest_float('gamma_value', 1e-5, 10, log=True)
            
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 1, 5)
            params['coef0'] = trial.suggest_float('coef0_poly', 0, 10)
        elif params['kernel'] == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0_sig', 0, 10)
            
        params.pop('gamma_type', None)
            
        svm = SVC(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(svm, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    bp = study.best_params
    best_params = {
        'C': bp['C'],
        'kernel': bp['kernel'],
        'shrinking': bp['shrinking'],
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    if bp['gamma_type'] == 'custom':
        best_params['gamma'] = bp['gamma_value']
    else:
        best_params['gamma'] = bp['gamma_type']
        
    if bp['kernel'] == 'poly':
        best_params['degree'] = bp['degree']
        best_params['coef0'] = bp['coef0_poly']
    elif bp['kernel'] == 'sigmoid':
        best_params['coef0'] = bp['coef0_sig']
        
    best_model = SVC(**best_params)
    best_model.fit(X_train, y_train)
    return best_model




def train_svm(X_train, y_train):
    """Train SVM Classifier with default parameters."""
    print("\n" + "="*60)
    print("Training SVM Classifier (default parameters)...")
    print("="*60)

    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    print("SVM training completed.")
    return svm_model


