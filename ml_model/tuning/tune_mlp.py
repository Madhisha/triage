import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform, loguniform

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.neural_network import MLPClassifier

def tune_mlp_random(X_train, y_train, n_iter=100):
    """Tune MLP using RandomizedSearchCV with distribution sampling."""
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
        'alpha': loguniform(1e-6, 1e-1),
        'batch_size': [16, 32, 64, 128, 256, 'auto'],             # 6 options
        'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 3 options
        'learning_rate_init': loguniform(1e-5, 1e-1),
        'max_iter': [200, 500, 1000, 2000],                       # 4 options
        'early_stopping': [True, False],                          # 2 options
        'validation_fraction': uniform(0.05, 0.15),
        'n_iter_no_change': randint(5, 31),
        'tol': loguniform(1e-5, 1e-2),
        'momentum': uniform(0.0, 1.0),
        'power_t': uniform(0.1, 0.9)
    }
    
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
    """Tune MLP using GridSearchCV with expanded grid."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: MLP (GridSearchCV - Massive)")
    print("="*60)
    
    param_grid = {
        'hidden_layer_sizes': [
            (128,), (256,), (512,), (1024,),
            (128, 64), (256, 128), (512, 256), (1024, 512),
            (256, 128, 64), (512, 256, 128), (1024, 512, 256)
        ],
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'learning_rate_init': [1e-5, 1e-4, 1e-3, 1e-2],
        'max_iter': [200, 500, 1000, 2000],
        'batch_size': [16, 32, 64, 128],
        'early_stopping': [True, False]
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


def tune_mlp_bayesian(X_train, y_train, n_trials=100):
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

