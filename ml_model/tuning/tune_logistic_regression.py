from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.linear_model import LogisticRegression

def tune_logistic_regression_random(X_train, y_train, n_iter=20):
    """Tune Logistic Regression using RandomizedSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (RandomizedSearchCV)")
    print("="*60)

    param_distributions = [
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.2, 0.5, 0.8],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        }
    ]

    lr_base = LogisticRegression(class_weight='balanced', random_state=42)

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    random_search = RandomizedSearchCV(
        lr_base,
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


def tune_logistic_regression_grid(X_train, y_train):
    """Tune Logistic Regression using GridSearchCV"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (GridSearchCV)")
    print("="*60)
    
    param_grid = [
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.2, 0.5, 0.8],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        },
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [1000, 2000],
            'fit_intercept': [True, False],
            'class_weight': ['balanced']
        }
    ]
    
    lr_base = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    
    grid_search = GridSearchCV(
        lr_base,
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


def tune_logistic_regression_bayesian(X_train, y_train, n_trials=30):
    """Tune Logistic Regression using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Falling back to RandomizedSearchCV...")
        return tune_logistic_regression_random(X_train, y_train)
        
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (Bayesian - Optuna)")
    print("="*60)
    
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none'])
        multi_class = trial.suggest_categorical('multi_class', ['ovr', 'multinomial'])
        
        if penalty == 'l1':
            solver = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
            if solver == 'liblinear' and multi_class == 'multinomial':
                return 0.0
            C = trial.suggest_float('C_l1', 0.01, 10.0, log=True)
            params = {'penalty': 'l1', 'solver': solver, 'C': C}
        elif penalty == 'l2':
            solver = trial.suggest_categorical('solver_l2', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            C = trial.suggest_float('C_l2', 0.01, 10.0, log=True)
            params = {'penalty': 'l2', 'solver': solver, 'C': C}
        elif penalty == 'elasticnet':
            params = {
                'penalty': 'elasticnet',
                'solver': 'saga',
                'C': trial.suggest_float('C_en', 0.01, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
            }
        else: # none
            solver = trial.suggest_categorical('solver_none', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            params = {'penalty': None, 'solver': solver}
            
        params.update({
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': trial.suggest_categorical('max_iter', [1000, 2000]),
            'multi_class': multi_class
        })
            
        lr = LogisticRegression(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(lr, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    bp = study.best_params
    penalty = bp['penalty']
    best_params = {'penalty': penalty if penalty != 'none' else None}
    
    if penalty == 'l1':
        best_params.update({'solver': bp['solver_l1'], 'C': bp['C_l1']})
    elif penalty == 'l2':
        best_params.update({'solver': bp['solver_l2'], 'C': bp['C_l2']})
    elif penalty == 'elasticnet':
        best_params.update({'solver': 'saga', 'C': bp['C_en'], 'l1_ratio': bp['l1_ratio']})
    else:
        best_params.update({'solver': bp['solver_none']})
        
    best_params.update({
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': bp['max_iter'],
        'multi_class': bp['multi_class']
    })
    
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)
    return best_model




def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression using sklearn defaults."""
    print("\n" + "="*60)
    print("Training Logistic Regression (default parameters)...")
    print("="*60)
    print("Note: Using LogisticRegression() defaults.")

    lr_model = LogisticRegression()

    lr_model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return lr_model






