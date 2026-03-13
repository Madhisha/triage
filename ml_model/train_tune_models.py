import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import joblib
import os
import numpy as np
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

def load_data():
    """Load the preprocessed train, validation, and test datasets with chief complaint features"""
    train_df = pd.read_csv("ml_processed_data/ml_processed_train.csv")
    valid_df = pd.read_csv("ml_processed_data/ml_processed_valid.csv")
    test_df = pd.read_csv("ml_processed_data/ml_processed_test.csv")

    # train_df = pd.read_csv("ml_processed_data/balanced/ml_processed_train.csv")
    # valid_df = pd.read_csv("ml_processed_data/balanced/ml_processed_valid.csv")
    # test_df = pd.read_csv("ml_processed_data/balanced/ml_processed_test.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {valid_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Count feature types
    tfidf_cols = [col for col in train_df.columns if col.startswith('tfidf_')]
    numeric_cols = [col for col in train_df.columns if not col.startswith('tfidf_') and col != 'acuity']
    
    print(f"\nFeature breakdown:")
    print(f"  - TF-IDF (Chief Complaint) features: {len(tfidf_cols)}")
    print(f"  - Physiological features: {len(numeric_cols)}")
    print(f"  - Total features: {len(tfidf_cols) + len(numeric_cols)}")
    
    return train_df, valid_df, test_df

def prepare_features_target(df, target_col='acuity', merge_classes=False):
    """Separate features and target variable"""
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # Merge classes 3, 4, 5 into class 3 (high acuity)
    if merge_classes:
        y = y.replace({4.0: 3.0, 5.0: 3.0})
        print("Merged classes 4 and 5 into class 3.")
        print(f"New class distribution:\n{y.value_counts().sort_index()}")
    
    return X, y

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
        max_features=max_features,  # Tunable parameter
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return rf_model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression (works well with TF-IDF features)"""
    print("\n" + "="*60)
    print("Training Logistic Regression (with Chief Complaint)...")
    print("="*60)
    print("Note: LR is effective for text classification tasks.")
    
    lr_model = LogisticRegression(
        max_iter=2000,  # Increased for convergence with more features
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        C=1.0,  # Regularization strength
        random_state=42,
        verbose=1
    )
    
    lr_model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return lr_model

def train_xgboost(X_train, y_train, n_estimators=1000):
    """Train XGBoost Classifier (handles sparse features well)"""
    print("\n" + "="*60)
    print("Training XGBoost Classifier (with Chief Complaint)...")
    print("="*60)
    print(f"Note: Using n_estimators={n_estimators}")
    
    # Convert labels to 0-indexed for XGBoost (1,2,3,4,5 -> 0,1,2,3,4)
    y_train_xgb = y_train - 1
    
    # Compute class weights for XGBoost
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,  # Increased for complex patterns
        max_depth=12,  # Moderate depth for text features
        learning_rate=0.05,  # Lower for better generalization
        subsample=0.8,
        colsample_bytree=0.6,  # Reduced to handle many features
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print("XGBoost training completed.")
    return xgb_model

def train_mlp(X_train, y_train):
    """Train Multi-Layer Perceptron (good for text + numeric features)"""
    print("\n" + "="*60)
    print("Training MLP Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: Neural networks can learn complex text patterns.")
    print("Using early stopping to prevent overfitting.")
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # Larger layers for more features
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
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

def train_lightgbm(X_train, y_train):
    """Train LightGBM Classifier (fast and efficient for large datasets)"""
    print("\n" + "="*60)
    print("Training LightGBM Classifier (with Chief Complaint)...")
    print("="*60)
    print("Note: LightGBM is very fast and memory efficient.")
    
    # Convert labels to 0-indexed for LightGBM
    y_train_lgb = y_train - 1
    
    # Compute class weights
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        num_leaves=20,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print("LightGBM training completed.")
    return lgb_model

def train_catboost(X_train, y_train):
    """Train CatBoost Classifier (handles categorical and text features natively)"""
    print("\n" + "="*60)
    print("Training CatBoost Classifier (with Chief Complaint)...")
    print("="*60)
    
    y_train_cat = y_train - 1
    classes = np.unique(y_train_cat)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cat)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        class_weights=class_weight_dict,
        random_seed=42,
        verbose=100,
        loss_function='MultiClass'
    )
    
    cat_model.fit(X_train, y_train_cat)
    print("CatBoost training completed.")
    return cat_model

def train_adaboost(X_train, y_train):
    """Train AdaBoost Classifier"""
    print("\n" + "="*60)
    print("Training AdaBoost Classifier...")
    print("="*60)
    print("Note: AdaBoost focuses on misclassified samples.")
    
    ada_model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    
    ada_model.fit(X_train, y_train)
    print("AdaBoost training completed.")
    return ada_model

def tune_random_forest_random(X_train, y_train, n_iter=100):
    """Tune Random Forest using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Random Forest (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = {
        'n_estimators': np.arange(100, 3001, 100).tolist(),
        'max_depth': [None] + np.arange(5, 51, 5).tolist(),
        'min_samples_split': np.arange(2, 21, 2).tolist(),
        'min_samples_leaf': np.arange(1, 11, 1).tolist(),
        'max_features': ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7, 0.9, None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy', 'log_loss']
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

# ==================== XGBoost Tuning ====================

def tune_xgboost_random(X_train, y_train, n_iter=100):
    """Tune XGBoost using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (RandomizedSearchCV - Massive)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_distributions = {
        'n_estimators': np.arange(100, 5001, 100).tolist(),
        'max_depth': np.arange(2, 21, 1).tolist(),
        'learning_rate': np.logspace(-4, -0.3, 50).tolist(),
        'subsample': np.arange(0.1, 1.05, 0.05).tolist(),
        'colsample_bytree': np.arange(0.1, 1.05, 0.05).tolist(),
        'colsample_bylevel': np.arange(0.1, 1.05, 0.05).tolist(),
        'min_child_weight': np.arange(1, 31, 1).tolist(),
        'gamma': np.arange(0, 10.1, 0.1).tolist(),
        'reg_alpha': np.logspace(-8, 1, 20).tolist(),
        'reg_lambda': np.logspace(-8, 1, 20).tolist(),
        'scale_pos_weight': [1, 2, 5, 10],  # Added for imbalance handling
        'booster': ['gbtree', 'dart']
    }
    
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_xgboost_grid(X_train, y_train):
    """Tune XGBoost using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (GridSearchCV - Massive)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    param_grid = {
        'n_estimators': [500, 1000, 2000, 3000],
        'max_depth': [4, 6, 8, 10, 14, 20],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_xgboost_bayesian(X_train, y_train, n_trials=30):
    """Tune XGBoost using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_xgboost_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: XGBoost (Bayesian - Optuna)")
    print("="*60)
    
    y_train_xgb = y_train - 1
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'gamma': trial.suggest_float('gamma', 0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        
        xgb = XGBClassifier(**params)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(xgb, X_train, y_train_xgb, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'mlogloss'
    
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train_xgb, sample_weight=sample_weights)
    
    return best_model

# ==================== MLP Tuning ====================

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
        'hidden_layer_sizes': layer_options,
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': np.logspace(-6, -1, 30).tolist(),
        'batch_size': [16, 32, 64, 128, 256, 'auto'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': np.logspace(-5, -1, 30).tolist(),
        'max_iter': [200, 500, 1000, 2000],
        'early_stopping': [True, False],
        'validation_fraction': np.linspace(0.05, 0.2, 5).tolist(),
        'n_iter_no_change': np.arange(5, 31, 5).tolist(),
        'tol': np.logspace(-5, -2, 10).tolist(),
        'momentum': np.linspace(0.0, 1.0, 11).tolist(),
        'power_t': np.linspace(0.1, 1.0, 10).tolist()
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

# ==================== LightGBM Tuning ====================

def tune_lightgbm_random(X_train, y_train, n_iter=100):
    """Tune LightGBM using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (RandomizedSearchCV - Massive)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    param_distributions = {
        'n_estimators': np.arange(100, 5001, 100).tolist(),
        'max_depth': [-1] + np.arange(5, 51, 5).tolist(),
        'learning_rate': np.logspace(-4, -0.3, 50).tolist(),
        'num_leaves': np.arange(20, 501, 20).tolist(),
        'min_child_samples': np.arange(5, 101, 5).tolist(),
        'subsample': np.arange(0.1, 1.05, 0.05).tolist(),
        'colsample_bytree': np.arange(0.1, 1.05, 0.05).tolist(),
        'reg_alpha': np.logspace(-8, 1, 20).tolist(),
        'reg_lambda': np.logspace(-8, 1, 20).tolist(),
        'boosting_type': ['gbdt', 'dart']
    }
    
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    random_search.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_lightgbm_grid(X_train, y_train):
    """Tune LightGBM using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (GridSearchCV - Massive)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    # Reduced grid for GridSearch
    param_grid = {
        'n_estimators': [500, 1000, 2000, 3000],
        'max_depth': [-1, 10, 20, 30, 50],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'boosting_type': ['gbdt', 'dart']
    }
    
    lgb_base = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    grid_search = GridSearchCV(
        lgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def tune_lightgbm_bayesian(X_train, y_train, n_trials=30):
    """Tune LightGBM using Bayesian Optimization (Optuna)"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to RandomizedSearchCV...")
        return tune_lightgbm_random(X_train, y_train, n_iter=20)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning: LightGBM (Bayesian - Optuna)")
    print("="*60)
    
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
            'max_depth': trial.suggest_categorical('max_depth', [-1] + list(range(5, 51, 5))),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        lgb = LGBMClassifier(**params)
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(lgb, X_train, y_train_lgb, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train_lgb, sample_weight=sample_weights)
    
    return best_model


# ==================== CatBoost Tuning ====================

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
        'iterations': np.arange(100, 3001, 100).tolist(),
        'depth': np.arange(2, 13, 1).tolist(),
        'learning_rate': np.logspace(-3, -0.3, 30).tolist(),
        'l2_leaf_reg': np.logspace(0, 2, 20).tolist(),
        'border_count': [32, 64, 128, 255],
        'thread_count': [-1],
        'random_seed': [42]
    }
    
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


# ==================== AdaBoost Tuning ====================

def tune_adaboost_random(X_train, y_train, n_iter=50):
    """Tune AdaBoost using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: AdaBoost (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = {
        'n_estimators': np.arange(50, 3001, 50).tolist(),
        'learning_rate': np.logspace(-3, 0.3, 30).tolist(),
        'algorithm': ['SAMME', 'SAMME.R']
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
    """Tune AdaBoost using GridSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: AdaBoost (GridSearchCV - Massive)")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 500, 1000, 2000, 3000],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0, 1.5],
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

def tune_adaboost_bayesian(X_train, y_train, n_trials=50):
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

# ==================== Logistic Regression Tuning ====================

def tune_logistic_regression_random(X_train, y_train, n_iter=50):
    """Tune Logistic Regression using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = [
        # l1 penalty solvers
        {
            'C': np.logspace(-5, 2, 40).tolist(),
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [2000]
        },
        # l2 penalty solvers
        {
            'C': np.logspace(-5, 2, 40).tolist(),
            'penalty': ['l2'],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [2000]
        },
        # elasticnet penalty (saga only)
        {
            'C': np.logspace(-5, 2, 40).tolist(),
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': np.linspace(0, 1, 11).tolist(),
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [2000]
        },
        # No penalty
        {
            'penalty': [None],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'multi_class': ['ovr', 'multinomial'],
            'max_iter': [2000]
        }
    ]
    
    lr_base = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Filter combinations (multinomial not supported by liblinear)
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
    """Tune Logistic Regression using GridSearchCV with moderate grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: Logistic Regression (GridSearchCV - Moderate)")
    print("="*60)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'multi_class': ['ovr', 'multinomial'],
        'max_iter': [1000, 2000]
    }
    
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

def tune_logistic_regression_bayesian(X_train, y_train, n_trials=50):
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

# ==================== SVM Tuning ====================

def tune_svm_random(X_train, y_train, n_iter=30):
    """Tune SVM using RandomizedSearchCV with massive grid"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning: SVM (RandomizedSearchCV - Massive)")
    print("="*60)
    
    param_distributions = {
        'C': np.logspace(-5, 2, 40).tolist(),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + np.logspace(-5, 1, 20).tolist(),
        'degree': np.arange(1, 6, 1).tolist(),
        'coef0': np.linspace(0, 10, 20).tolist(),
        'shrinking': [True, False],
        'probability': [True]
    }
    
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

def evaluate_model(model, X, y, dataset_name="Dataset", is_xgb=False, is_catboost=False, is_lightgbm=False, output_file=None):
    """Evaluate model performance with accuracy, classification report, AUROC, and AUPRC"""
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_name}")
    print("="*60)
    
    y_pred = model.predict(X)
    
    # Convert predictions back to original labels (0,1,2 -> 1,2,3)
    if is_xgb or is_catboost or is_lightgbm:
        y_pred = y_pred + 1
    
    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compute AUROC and AUPRC
    try:
        # Get probability predictions
        y_proba = model.predict_proba(X)
        
        # Adjust probabilities for XGBoost/CatBoost/LightGBM (0-indexed)
        if is_xgb or is_catboost or is_lightgbm:
            # For 0-indexed models, probabilities are already aligned with classes 0,1,2
            # But we need to align them with true labels 1,2,3
            classes = np.array([1, 2, 3])  # True class labels
        else:
            classes = model.classes_
        
        # Binarize the labels for multi-class ROC/PR computation
        y_bin = label_binarize(y, classes=classes)
        
        # Handle binary classification case (returns 1D array)
        if len(classes) == 2:
            y_bin = np.hstack([1 - y_proba[:, 1:2], y_proba[:, 1:2]])
        
        # Compute macro-averaged AUROC and AUPRC
        auroc_macro = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
        auprc_macro = average_precision_score(y_bin, y_proba, average='macro')
        
        # Compute weighted-averaged AUROC and AUPRC
        auroc_weighted = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        auprc_weighted = average_precision_score(y_bin, y_proba, average='weighted')
        
        # Compute per-class AUROC and AUPRC
        auroc_per_class = roc_auc_score(y_bin, y_proba, average=None, multi_class='ovr')
        auprc_per_class = average_precision_score(y_bin, y_proba, average=None)
        
        print(f"\nAUROC (macro): {auroc_macro:.4f}")
        print(f"AUROC (weighted): {auroc_weighted:.4f}")
        print(f"AUPRC (macro): {auprc_macro:.4f}")
        print(f"AUPRC (weighted): {auprc_weighted:.4f}")
        
        print("\nPer-class AUROC:")
        for i, cls in enumerate(classes):
            print(f"  Class {cls}: {auroc_per_class[i]:.4f}")
        
        print("\nPer-class AUPRC:")
        for i, cls in enumerate(classes):
            print(f"  Class {cls}: {auprc_per_class[i]:.4f}")
        
    except Exception as e:
        print(f"\nWarning: Could not compute AUROC/AUPRC: {e}")
        auroc_macro = auroc_weighted = auprc_macro = auprc_weighted = None
        auroc_per_class = auprc_per_class = None
    
    # Classification Report
    report = classification_report(y, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Save to file if provided
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"Evaluation on {dataset_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            
            if auroc_macro is not None:
                f.write(f"\nAUROC (macro): {auroc_macro:.4f}\n")
                f.write(f"AUROC (weighted): {auroc_weighted:.4f}\n")
                f.write(f"AUPRC (macro): {auprc_macro:.4f}\n")
                f.write(f"AUPRC (weighted): {auprc_weighted:.4f}\n")
                
                f.write("\nPer-class AUROC:\n")
                for i, cls in enumerate(classes):
                    f.write(f"  Class {cls}: {auroc_per_class[i]:.4f}\n")
                
                f.write("\nPer-class AUPRC:\n")
                for i, cls in enumerate(classes):
                    f.write(f"  Class {cls}: {auprc_per_class[i]:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write(report + "\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm) + "\n")
    
    return accuracy, y_pred

def show_feature_importance(model, feature_names, top_n=20):
    """Show feature importance for tree-based models, highlighting text vs numeric"""
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*60)
        print(f"Top {top_n} Most Important Features:")
        print("="*60)
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Type': ['Text' if str(f).startswith('tfidf_') else 'Numeric' for f in feature_names]
        }).sort_values(by='Importance', ascending=False).head(top_n)
        
        print(feature_importance_df.to_string(index=False))
        
        # Summary statistics
        text_features = feature_importance_df[feature_importance_df['Type'] == 'Text'].shape[0]
        numeric_features = top_n - text_features
        print(f"\nTop {top_n} features: {text_features} Text, {numeric_features} Numeric")
        
        # Total importance by type
        all_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Type': ['Text' if str(f).startswith('tfidf_') else 'Numeric' for f in feature_names]
        })
        type_importance = all_importance_df.groupby('Type')['Importance'].sum()
        print(f"\nTotal importance by type:")
        for ftype, imp in type_importance.items():
            pct = (imp / importances.sum()) * 100
            print(f"  {ftype}: {imp:.4f} ({pct:.1f}%)")

def save_model(model, filename):
    """Save trained model to disk"""
    models_dir = "ml_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")

def main():
    print("Loading data...")
    train_df, valid_df, test_df = load_data()
    
    # Create results output file
    results_file = "training_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MODEL TRAINING AND EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
    
    # Prepare data
    X_train, y_train = prepare_features_target(train_df)
    X_valid, y_valid = prepare_features_target(valid_df)
    X_test, y_test = prepare_features_target(test_df)
    
    print(f"\nTotal features: {len(X_train.columns)}")
    tfidf_features = [f for f in X_train.columns if str(f).startswith('tfidf_')]
    print(f"Sample TF-IDF features: {tfidf_features[:5]}")
    
    # Ask user if they want to merge classes
    print("\nMerge acuity classes 3, 4, 5 into a single class?")
    print("(This can improve accuracy due to severe class imbalance)")
    print("1. Yes - Merge to 3 classes (1, 2, 3)")
    print("2. No - Keep original 5 classes")
    merge_choice = input("Enter choice (1 or 2): ").strip()
    
    merge_classes = merge_choice == '1'
    
    if merge_classes:
        # Re-prepare with merged classes
        X_train, y_train = prepare_features_target(train_df, merge_classes=True)
        X_valid, y_valid = prepare_features_target(valid_df, merge_classes=True)
        X_test, y_test = prepare_features_target(test_df, merge_classes=True)
    
    print(f"\nTarget distribution in training set:")
    print(y_train.value_counts().sort_index())
    
    # Ask user for model choice
    print("\nChoose model(s) to train:")
    print("1. Random Forest")
    print("2. Logistic Regression")
    print("3. XGBoost")
    print("4. MLP (Neural Network)")
    print("5. CatBoost")
    print("6. LightGBM")
    print("7. AdaBoost")
    print("8. SVM (Support Vector Machine)")
    print("9. All models (EXCEPT SVM)")
    print("10. All models (INCLUDING SVM)")
    choice = input("Enter choice (1-10): ").strip()
    
    do_train_rf = choice in ['1', '9', '10']
    do_train_lr = choice in ['2', '9', '10']
    do_train_xgb = choice in ['3', '9', '10']
    do_train_mlp = choice in ['4', '9', '10']
    do_train_catboost = choice in ['5', '9', '10']
    do_train_lightgbm = choice in ['6', '9', '10']
    do_train_adaboost = choice in ['7', '9', '10']
    do_train_svm = choice in ['8', '10']
    
    if choice == '9':
        do_train_rf = do_train_lr = do_train_xgb = do_train_mlp = do_train_catboost = do_train_lightgbm = do_train_adaboost = True
        do_train_svm = False
    elif choice == '10':
        do_train_rf = do_train_lr = do_train_xgb = do_train_mlp = do_train_catboost = do_train_lightgbm = do_train_adaboost = do_train_svm = True

    # Global tuning choice for "All" options
    global_tuning = None
    if choice in ['9', '10']:
        print("\nYou have selected all models. Would you like to set a GLOBAL tuning method for all of them?")
        print("1. Set a global tuning method for all selected models")
        print("2. Choose tuning method for each model individually")
        global_choice = input("Enter choice (1 or 2): ").strip()
        
        if global_choice == '1':
            print("\nSelect GLOBAL Hyperparameter Tuning Method:")
            print("1. No tuning (use default parameters)")
            print("2. RandomizedSearchCV (Massive range)")
            print("3. GridSearchCV (Restricted range for speed)")
            print("4. Bayesian Optimization (Optuna - Wide range)")
            gtc = input("Enter choice (1-4): ").strip()
            global_tuning = {'2': 'random', '3': 'grid', '4': 'bayesian'}.get(gtc, 'none')
            if gtc == '1': global_tuning = 'none'

    # Helper for generic tuning prompt
    def get_tuning_choice(model_name):
        if global_tuning:
            return None if global_tuning == 'none' else global_tuning
            
        print(f"\nHyperparameter Tuning for {model_name}:")
        print("1. No tuning (use default parameters)")
        print("2. RandomizedSearchCV (Massive range)")
        print("3. GridSearchCV (Restricted range for speed)")
        print("4. Bayesian Optimization (Optuna - Wide range)")
        tc = input("Enter choice (1-4): ").strip()
        return {'2': 'random', '3': 'grid', '4': 'bayesian'}.get(tc, None)

    rf_tuning_method = get_tuning_choice("Random Forest") if do_train_rf else None
    lr_tuning_method = get_tuning_choice("Logistic Regression") if do_train_lr else None
    xgb_tuning_method = get_tuning_choice("XGBoost") if do_train_xgb else None
    mlp_tuning_method = get_tuning_choice("MLP") if do_train_mlp else None
    cat_tuning_method = get_tuning_choice("CatBoost") if do_train_catboost else None
    lgb_tuning_method = get_tuning_choice("LightGBM") if do_train_lightgbm else None
    ada_tuning_method = get_tuning_choice("AdaBoost") if do_train_adaboost else None
    svm_tuning_method = get_tuning_choice("SVM") if do_train_svm else None
    
    rf_model = None
    lr_model = None
    xgb_model = None
    mlp_model = None
    catboost_model = None
    lightgbm_model = None
    adaboost_model = None
    svm_model = None
    
    # Train Random Forest
    if do_train_rf:
        if rf_tuning_method == 'random':
            rf_model = tune_random_forest_random(X_train, y_train, n_iter=100)
        elif rf_tuning_method == 'grid':
            rf_model = tune_random_forest_grid(X_train, y_train)
        elif rf_tuning_method == 'bayesian':
            rf_model = tune_random_forest_bayesian(X_train, y_train, n_trials=50)
        else:
            rf_model = train_random_forest(X_train, y_train)
        
        evaluate_model(rf_model, X_valid, y_valid, "Validation Set (Random Forest)", output_file=results_file)
        show_feature_importance(rf_model, X_train.columns)
    
    # Train Logistic Regression
    if do_train_lr:
        if lr_tuning_method == 'random':
            lr_model = tune_logistic_regression_random(X_train, y_train)
        elif lr_tuning_method == 'grid':
            lr_model = tune_logistic_regression_grid(X_train, y_train)
        elif lr_tuning_method == 'bayesian':
            lr_model = tune_logistic_regression_bayesian(X_train, y_train)
        else:
            lr_model = train_logistic_regression(X_train, y_train)
        evaluate_model(lr_model, X_valid, y_valid, "Validation Set (Logistic Regression)", output_file=results_file)

    # Train XGBoost
    if do_train_xgb:
        if xgb_tuning_method == 'random':
            xgb_model = tune_xgboost_random(X_train, y_train, n_iter=100)
        elif xgb_tuning_method == 'grid':
            xgb_model = tune_xgboost_grid(X_train, y_train)
        elif xgb_tuning_method == 'bayesian':
            xgb_model = tune_xgboost_bayesian(X_train, y_train, n_trials=100)
        else:
            xgb_model = train_xgboost(X_train, y_train)
        
        evaluate_model(xgb_model, X_valid, y_valid, "Validation Set (XGBoost)", is_xgb=True, output_file=results_file)
        show_feature_importance(xgb_model, X_train.columns)
    
    # Train MLP
    if do_train_mlp:
        if mlp_tuning_method == 'random':
            mlp_model = tune_mlp_random(X_train, y_train, n_iter=100)
        elif mlp_tuning_method == 'grid':
            mlp_model = tune_mlp_grid(X_train, y_train)
        elif mlp_tuning_method == 'bayesian':
            mlp_model = tune_mlp_bayesian(X_train, y_train, n_trials=50)
        else:
            mlp_model = train_mlp(X_train, y_train)
        
        evaluate_model(mlp_model, X_valid, y_valid, "Validation Set (MLP)", output_file=results_file)
        
    # Train CatBoost
    if do_train_catboost:
        if cat_tuning_method == 'random':
            catboost_model = tune_catboost_random(X_train, y_train)
        elif cat_tuning_method == 'grid':
            catboost_model = tune_catboost_grid(X_train, y_train)
        elif cat_tuning_method == 'bayesian':
            catboost_model = tune_catboost_bayesian(X_train, y_train)
        else:
            catboost_model = train_catboost(X_train, y_train)
        evaluate_model(catboost_model, X_valid, y_valid, "Validation Set (CatBoost)", is_catboost=True, output_file=results_file)
        show_feature_importance(catboost_model, X_train.columns)
    
    # Train LightGBM
    if do_train_lightgbm:
        if lgb_tuning_method == 'random':
            lightgbm_model = tune_lightgbm_random(X_train, y_train, n_iter=100)
        elif lgb_tuning_method == 'grid':
            lightgbm_model = tune_lightgbm_grid(X_train, y_train)
        elif lgb_tuning_method == 'bayesian':
            lightgbm_model = tune_lightgbm_bayesian(X_train, y_train, n_trials=100)
        else:
            lightgbm_model = train_lightgbm(X_train, y_train)
        
        evaluate_model(lightgbm_model, X_valid, y_valid, "Validation Set (LightGBM)", is_lightgbm=True, output_file=results_file)
        show_feature_importance(lightgbm_model, X_train.columns)
    
    # Train AdaBoost
    if do_train_adaboost:
        if ada_tuning_method == 'random':
            adaboost_model = tune_adaboost_random(X_train, y_train)
        elif ada_tuning_method == 'grid':
            adaboost_model = tune_adaboost_grid(X_train, y_train)
        elif ada_tuning_method == 'bayesian':
            adaboost_model = tune_adaboost_bayesian(X_train, y_train)
        else:
            adaboost_model = train_adaboost(X_train, y_train)
        evaluate_model(adaboost_model, X_valid, y_valid, "Validation Set (AdaBoost)", output_file=results_file)

    # Train SVM
    if do_train_svm:
        if svm_tuning_method == 'random':
            svm_model = tune_svm_random(X_train, y_train)
        elif svm_tuning_method == 'grid':
            svm_model = tune_svm_grid(X_train, y_train)
        elif svm_tuning_method == 'bayesian':
            svm_model = tune_svm_bayesian(X_train, y_train)
        else:
            svm_model = SVC(class_weight='balanced', probability=True, random_state=42)
            print("\nTraining SVM with default parameters...")
            svm_model.fit(X_train, y_train)
        evaluate_model(svm_model, X_valid, y_valid, "Validation Set (SVM)", output_file=results_file)
    
    # Final evaluation on test set
    print("\n" + "#"*60)
    print("FINAL EVALUATION ON TEST SET")
    print("#"*60)
    
    if rf_model:
        print("\nRandom Forest on Test Set:")
        evaluate_model(rf_model, X_test, y_test, "Test Set (Random Forest)", output_file=results_file)
    
    if lr_model:
        print("\nLogistic Regression on Test Set:")
        evaluate_model(lr_model, X_test, y_test, "Test Set (Logistic Regression)", output_file=results_file)
    
    if xgb_model:
        print("\nXGBoost on Test Set:")
        evaluate_model(xgb_model, X_test, y_test, "Test Set (XGBoost)", is_xgb=True, output_file=results_file)
    
    if mlp_model:
        print("\nMLP on Test Set:")
        evaluate_model(mlp_model, X_test, y_test, "Test Set (MLP)", output_file=results_file)
    
    if catboost_model:
        print("\nCatBoost on Test Set:")
        evaluate_model(catboost_model, X_test, y_test, "Test Set (CatBoost)", is_catboost=True, output_file=results_file)
    
    if lightgbm_model:
        print("\nLightGBM on Test Set:")
        evaluate_model(lightgbm_model, X_test, y_test, "Test Set (LightGBM)", is_lightgbm=True, output_file=results_file)
    
    if adaboost_model:
        print("\nAdaBoost on Test Set:")
        evaluate_model(adaboost_model, X_test, y_test, "Test Set (AdaBoost)", output_file=results_file)

    if svm_model:
        print("\nSVM on Test Set:")
        evaluate_model(svm_model, X_test, y_test, "Test Set (SVM)", output_file=results_file)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print("="*60)

if __name__ == "__main__":
    main()
