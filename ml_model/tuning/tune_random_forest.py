from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def tune_random_forest_random(X_train, y_train, n_iter=20):
    """Tune Random Forest using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (RandomizedSearchCV)")
    print("="*60)

    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 30, 50],
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
    """Tune Random Forest using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (GridSearchCV)")
    print("="*60)

    param_grid = {
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 30],
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 30, 50]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        rf = RandomForestClassifier(**params)

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")

    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model


def train_random_forest(X_train, y_train, max_features='log2', n_estimators=1000):
    """Train Random Forest Classifier (optimized for text + numeric features)"""
    print("\n" + "="*60)
    print("Training Random Forest Classifier (with Chief Complaint)...")
    print("="*60)
    print(f"Note: Using max_features='{max_features}', n_estimators={n_estimators}")

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=max_features,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return rf_model

