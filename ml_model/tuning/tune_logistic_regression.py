from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, loguniform, uniform

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.linear_model import LogisticRegression

def tune_logistic_regression_random(X_train, y_train, n_iter=50):
    """Tune Logistic Regression using RandomizedSearchCV with distribution sampling."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (RandomizedSearchCV - Massive)")
    print("="*60)
    
    # Separate grids to avoid invalid solver/multi_class combinations.
    param_distributions = [
        # l1 with liblinear (multinomial not supported by liblinear)
        {
            'C': loguniform(1e-5, 1e2),
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'max_iter': randint(500, 5001),
            'intercept_scaling': loguniform(5e-1, 5.0),
            'fit_intercept': [True, False],                       # 2
            'class_weight': ['balanced', None]                    # 2
        },
        # l1 with saga (supports multinomial)
        {
            'C': loguniform(1e-5, 1e2),
            'penalty': ['l1'],
            'solver': ['saga'],
            'multi_class': ['ovr', 'multinomial'],                # 2
            'max_iter': randint(500, 5001),
            'intercept_scaling': loguniform(5e-1, 5.0),
            'fit_intercept': [True, False],                       # 2
            'class_weight': ['balanced', None]                    # 2
        },
        # l2 penalty solvers
        {
            'C': loguniform(1e-5, 1e2),
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],                # 2
            'max_iter': randint(500, 5001),
            'intercept_scaling': loguniform(5e-1, 5.0),
            'fit_intercept': [True, False],                       # 2
            'class_weight': ['balanced', None]                    # 2
        },
        # elasticnet penalty (saga only)
        {
            'C': loguniform(1e-5, 1e2),
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': uniform(0.0, 1.0),
            'multi_class': ['ovr', 'multinomial'],                # 2
            'max_iter': randint(500, 5001),
            'intercept_scaling': loguniform(5e-1, 5.0),
            'fit_intercept': [True, False],                       # 2
            'class_weight': ['balanced', None]                    # 2
        },
        # No penalty
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],                # 2
            'max_iter': randint(500, 5001),
            'intercept_scaling': loguniform(5e-1, 5.0),
            'fit_intercept': [True, False],                       # 2
            'class_weight': ['balanced', None]                    # 2
        }
    ]
    
    lr_base = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Filter warnings from incompatible combinations that can still occur during random sampling.
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
    """Tune Logistic Regression using GridSearchCV with expanded valid grids."""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (GridSearchCV - Moderate)")
    print("="*60)
    
    param_grid = [
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'max_iter': [500, 1000, 2000, 3000, 5000]
        },
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            'penalty': ['l1'],
            'solver': ['saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [500, 1000, 2000, 3000, 5000]
        },
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [500, 1000, 2000, 3000, 5000]
        },
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.0, 0.2, 0.5, 0.8, 1.0],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [500, 1000, 2000, 3000, 5000]
        },
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [500, 1000, 2000, 3000, 5000]
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


def tune_logistic_regression_bayesian(X_train, y_train, n_trials=100):
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
            C = trial.suggest_float('C_l1', 1e-5, 100, log=True)
            params = {'penalty': 'l1', 'solver': solver, 'C': C}
        elif penalty == 'l2':
            solver = trial.suggest_categorical('solver_l2', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            C = trial.suggest_float('C_l2', 1e-5, 100, log=True)
            params = {'penalty': 'l2', 'solver': solver, 'C': C}
        elif penalty == 'elasticnet':
            params = {
                'penalty': 'elasticnet',
                'solver': 'saga',
                'C': trial.suggest_float('C_en', 1e-5, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
            }
        else: # none
            solver = trial.suggest_categorical('solver_none', ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            params = {'penalty': None, 'solver': solver}
            
        params.update({
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 2000,
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
        'max_iter': 2000,
        'multi_class': bp['multi_class']
    })
    
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)
    return best_model




def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with default parameters."""
    print("\n" + "="*60)
    print("Training Logistic Regression (default parameters)...")
    print("="*60)
    lr_model = LogisticRegression()
    
    lr_model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return lr_model






