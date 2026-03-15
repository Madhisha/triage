from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.neural_network import MLPClassifier

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
        """Train Multi-Layer Perceptron (good for text + numeric features)"""
    print("\n" + "="*60)
        print("Training MLP Classifier (with Chief Complaint)...")
    print("="*60)
        print("Note: Neural networks can learn complex text patterns.")
        print("Using early stopping to prevent overfitting.")

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
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

