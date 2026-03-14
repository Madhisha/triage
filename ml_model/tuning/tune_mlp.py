import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.neural_network import MLPClassifier

def tune_mlp_random(X_train, y_train, n_iter=100):
    """Tune MLP using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (RandomizedSearchCV - Massive)")
    print("="*60)
    
    # Granular layer sizes
    layer_options = [
        (128,), (256,), (512,), (1024,),
        (128, 64), (256, 128), (512, 256), (1024, 512),
        (256, 128, 64), (512, 256, 128), (1024, 512, 256),
        (256, 128, 64, 32), (512, 256, 128, 64)
    ]
    
    param_distributions = {
        'hidden_layer_sizes': layer_options,                      # 13 options
        'activation': ['relu', 'tanh', 'logistic', 'identity'],   # 4 options
        'solver': ['adam', 'sgd', 'lbfgs'],                       # 3 options
        'alpha': np.logspace(-6, -1, 12).tolist(),                # 12 options
        'batch_size': [16, 32, 64, 128, 256, 'auto'],             # 6 options
        'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 3 options
        'learning_rate_init': np.logspace(-5, -1, 12).tolist(),   # 12 options
        'max_iter': [200, 500, 1000, 2000],                       # 4 options
        'early_stopping': [True, False],                          # 2 options
        'validation_fraction': np.linspace(0.05, 0.2, 4).tolist(),# 4 options
        'n_iter_no_change': np.arange(5, 31, 5).tolist(),         # 6 options
        'tol': np.logspace(-5, -2, 6).tolist(),                   # 6 options
        'momentum': np.linspace(0.0, 1.0, 6).tolist(),            # 6 options
        'power_t': np.linspace(0.1, 1.0, 5).tolist()              # 5 options
    }
    # Total combinations: 13 * 4 * 3 * 12 * 6 * 3 * 12 * 4 * 2 * 4 * 6 * 6 * 6 * 5 = 2,090,188,800
    # ~2.09 Billion (safely below 2.14B max int32 value)
    
    mlp_base = MLPClassifier(
        random_state=42,
        verbose=False  # Set to False to keep CV output clean
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
    """Tune MLP using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (GridSearchCV - Massive)")
    print("="*60)
    
    # Note: Very reduced grid for GridSearch
    param_grid = {
        'hidden_layer_sizes': [(100,), (256,), (512,), (100, 50), (256, 128), (512, 256, 128)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.0001],
        'max_iter': [500, 1000],
        'batch_size': [32, 64],
        'early_stopping': [True]
    }
    
    mlp_base = MLPClassifier(
        random_state=42,
        verbose=False
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
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_categorical(f'n_units_l{i}', [32, 64, 128, 256, 512, 1024]))
        
        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic', 'identity']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 'auto']),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
            'max_iter': trial.suggest_categorical('max_iter', [200, 500, 1000, 2000]),
            'random_state': 42,
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.2),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 30),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'momentum': trial.suggest_float('momentum', 0.0, 1.0),
            'power_t': trial.suggest_float('power_t', 0.1, 1.0)
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




def train_mlp(X_train, y_train):
    """Train Multi-Layer Perceptron with default parameters."""
    print("\n" + "="*60)
    print("Training MLP Classifier (default parameters)...")
    print("="*60)
    mlp_model = MLPClassifier()
    
    mlp_model.fit(X_train, y_train)
    print("MLP training completed.")
    return mlp_model

