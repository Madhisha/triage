
"""
Ensemble Model Comparison
Base Models: MLP, XGBoost, Random Forest
Ensemble Types: Hard Voting, Soft Voting, Weighted Voting, Stacking
Optional Novelty: Stacking with Logistic Regression
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed data from ml_processed_data directory"""
    print("Loading preprocessed data...")
    
    train_df = pd.read_csv("ml_processed_data/ml_processed_train.csv")
    valid_df = pd.read_csv("ml_processed_data/ml_processed_valid.csv")
    test_df = pd.read_csv("ml_processed_data/ml_processed_test.csv")
    
    print(f"  Train: {train_df.shape}")
    print(f"  Valid: {valid_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    return train_df, valid_df, test_df


def merge_classes(y):
    """Merge classes 3, 4, 5 into class 3 for 3-class problem
    Class 1: Critical
    Class 2: Urgent  
    Class 3: Semi-urgent+ (merged 3, 4, 5)
    """
    y_merged = y.copy()
    y_merged = y_merged.replace({4: 3, 5: 3})
    return y_merged


def prepare_features(train_df, valid_df, test_df):
    """Prepare features and target variables"""
    
    # Target column
    target_col = 'acuity'
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Merge classes 3, 4, 5 into class 3
    print("\nMerging classes: 3, 4, 5 -> 3 (3-class problem)")
    y_train = merge_classes(y_train)
    y_valid = merge_classes(y_valid)
    y_test = merge_classes(y_test)
    
    print(f"Features shape: {X_train.shape[1]} columns")
    print(f"Class distribution (train): {y_train.value_counts().sort_index().to_dict()}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_random_forest(X_train, y_train):
    """Train Random Forest with optimized parameters"""
    print("\n  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',  
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    print("  ✅ Random Forest training completed")
    return rf_model


def train_xgboost(X_train, y_train):
    """Train XGBoost with optimized parameters"""
    print("\n  Training XGBoost...")
    
    # Convert labels to 0-indexed for XGBoost
    y_train_xgb = y_train - y_train.min()
    
    # Compute class weights
    classes = np.unique(y_train_xgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_xgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_xgb])
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,  # Increased for complex patterns
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
    print("  ✅ XGBoost training completed")
    return xgb_model


def train_mlp(X_train, y_train):
    """Train MLP with optimized parameters"""
    print("\n  Training MLP...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # Larger layers for more features
        activation='tanh',
        solver='adam',
        alpha=0.01,  # L2 regularization
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
    print("  ✅ MLP training completed")
    return mlp_model


def save_base_model(model, model_name, base_models_dir='base_models'):
    """Save a base model to disk"""
    os.makedirs(base_models_dir, exist_ok=True)
    filename = f'base_model_{model_name.lower().replace(" ", "_")}.pkl'
    filepath = os.path.join(base_models_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  💾 Saved: {model_name} to {filename}")


def load_or_train_base_models(X_train, y_train, X_valid, y_valid, base_models_dir='base_models'):
    """
    Load pre-trained base models from base_models directory.
    If models don't exist, train them and save them.
    
    Returns:
        dict: Dictionary of {model_name: model_object}
        dict: Dictionary of {model_name: {'accuracy': acc, 'f1': f1}}
    """
    print("=" * 70)
    print("LOADING OR TRAINING BASE MODELS")
    print("=" * 70)
    
    trained_models = {}
    results = {}
    y_train_min = y_train.min()
    
    model_info = {
        'Random Forest': ('base_model_random_forest.pkl', train_random_forest),
        'XGBoost': ('base_model_xgboost.pkl', train_xgboost),
        'MLP': ('base_model_mlp.pkl', train_mlp)
    }
    
    for model_name, (filename, train_func) in model_info.items():
        filepath = os.path.join(base_models_dir, filename)
        model_loaded = False
        
        if os.path.exists(filepath):
            try:
                # Load existing model
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                
                # Validate that it's a model with predict method
                if not hasattr(model, 'predict'):
                    print(f"⚠️  Invalid model file: {filename} (not a valid model object)")
                    raise ValueError("Not a valid model")
                
                trained_models[model_name] = model
                print(f"✅ Loaded: {model_name} from {filename}")
                model_loaded = True
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {str(e)}")
                print(f"   Will retrain {model_name}...")
        
        if not model_loaded:
            # Train new model
            if not os.path.exists(filepath):
                print(f"⚠️  Not found: {filename}")
            print(f"   Training {model_name}...")
            model = train_func(X_train, y_train)
            save_base_model(model, model_name, base_models_dir)
            trained_models[model_name] = model
        
        # Evaluate model on validation set
        if model_name == 'XGBoost':
            y_pred = model.predict(X_valid) + y_train_min
        else:
            y_pred = model.predict(X_valid)
        
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
        results[model_name] = {'accuracy': acc, 'f1': f1}
        print(f"  📊 {model_name}: Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")
    
    print(f"\n✅ Total base models ready: {len(trained_models)}")
    print()
    return trained_models, results


def save_ensemble_model(model, filename='stacking_ensemble_lr.pkl', output_dir='ensemble_model'):
    """
    Save the trained ensemble model to a file.
    
    Args:
        model: Trained ensemble model
        filename: Output filename
        output_dir: Directory to save the model (default: ensemble_model)
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  💾 Ensemble model saved to: {filepath}")


def load_ensemble_model(filename, output_dir='ensemble_model'):
    """Load a saved ensemble model if present; return None otherwise."""
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            # Basic sanity check
            if not hasattr(model, 'predict'):
                raise ValueError("Loaded object has no predict method")
            print(f"  ✅ Loaded ensemble model from: {filepath}")
            return model
        except Exception as e:
            print(f"  ⚠️  Could not load {filepath}: {e}. Will retrain.")
            return None
    return None


def hard_voting_ensemble(trained_models, X_valid, y_valid, y_train_min):
    """Hard Voting: Majority vote from base models"""
    
    predictions = {}
    for name, model in trained_models.items():
        if name == 'XGBoost':
            predictions[name] = model.predict(X_valid) + y_train_min
        else:
            predictions[name] = model.predict(X_valid)
    
    # Stack predictions and take mode (majority vote)
    pred_matrix = np.column_stack([predictions['Random Forest'], 
                                    predictions['XGBoost'], 
                                    predictions['MLP']])
    
    from scipy import stats
    hard_vote_pred, _ = stats.mode(pred_matrix, axis=1, keepdims=False)
    hard_vote_pred = hard_vote_pred.flatten()
    
    acc = accuracy_score(y_valid, hard_vote_pred)
    f1 = f1_score(y_valid, hard_vote_pred, average='macro', zero_division=0)
    
    return acc, f1, hard_vote_pred


def soft_voting_ensemble(trained_models, X_valid, y_valid, y_train_min):
    """Soft Voting: Average probability predictions"""
    
    probas = []
    classes = None
    
    for name, model in trained_models.items():
        proba = model.predict_proba(X_valid)
        if name == 'XGBoost':
            # XGBoost classes are 0-indexed
            if classes is None:
                classes = np.array(model.classes_) + y_train_min
        else:
            if classes is None:
                classes = model.classes_
        probas.append(proba)
    
    # Average probabilities
    avg_proba = np.mean(probas, axis=0)
    soft_vote_pred = classes[np.argmax(avg_proba, axis=1)]
    
    acc = accuracy_score(y_valid, soft_vote_pred)
    f1 = f1_score(y_valid, soft_vote_pred, average='macro', zero_division=0)
    
    return acc, f1, soft_vote_pred


def weighted_voting_ensemble(trained_models, X_valid, y_valid, y_train_min, weights=None):
    """Weighted Voting: Weighted average of probability predictions"""
    
    if weights is None:
        # Default weights based on typical model performance
        weights = {'Random Forest': 0.35, 'XGBoost': 0.40, 'MLP': 0.25}
    
    probas = {}
    classes = None
    
    for name, model in trained_models.items():
        proba = model.predict_proba(X_valid)
        if name == 'XGBoost':
            if classes is None:
                classes = np.array(model.classes_) + y_train_min
        else:
            if classes is None:
                classes = model.classes_
        probas[name] = proba
    
    # Weighted average of probabilities
    weighted_proba = sum(weights[name] * probas[name] for name in probas)
    weighted_vote_pred = classes[np.argmax(weighted_proba, axis=1)]
    
    acc = accuracy_score(y_valid, weighted_vote_pred)
    f1 = f1_score(y_valid, weighted_vote_pred, average='macro', zero_division=0)
    
    return acc, f1, weighted_vote_pred, weights


def find_best_weights(trained_models, X_valid, y_valid, y_train_min):
    """Grid search for optimal weights"""
    
    best_acc = 0
    best_weights = None
    
    for rf_w in np.arange(0.1, 0.6, 0.1):
        for xgb_w in np.arange(0.1, 0.6, 0.1):
            mlp_w = 1.0 - rf_w - xgb_w
            if mlp_w < 0.05:  # Skip invalid weights
                continue
            
            weights = {'Random Forest': rf_w, 'XGBoost': xgb_w, 'MLP': mlp_w}
            acc, _, _, _ = weighted_voting_ensemble(trained_models, X_valid, y_valid, y_train_min, weights)
            
            if acc > best_acc:
                best_acc = acc
                best_weights = weights
    
    return best_weights, best_acc


def stacking_ensemble(X_train, y_train, X_valid, y_valid, trained_models, final_estimator='rf'):
    """Stacking Ensemble using already-trained base models with specified final estimator"""
    
    # Use the already-trained models as base estimators
    estimators = [
        ('random_forest', trained_models['Random Forest']),
        ('xgboost', trained_models['XGBoost']),
        ('mlp', trained_models['MLP'])
    ]
    
    # Choose final estimator
    if final_estimator == 'lr':
        final_est = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
        name = "Stacking (Logistic Regression)"
    else:
        final_est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        name = "Stacking (Random Forest)"
    
    # Create stacking classifier with 5-fold CV
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    print(f"\n  Training {name}...")
    
    # For XGBoost in stacking, we need 0-indexed labels
    y_train_stacking = y_train - y_train.min()
    
    stacking_clf.fit(X_train, y_train_stacking)
    
    y_pred = stacking_clf.predict(X_valid) + y_train.min()
    
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
    
    return stacking_clf, acc, f1, name, y_pred


def main():
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL COMPARISON")
    print("Base Models: MLP, XGBoost, Random Forest")
    print("=" * 70)
    
    # Load and prepare data
    train_df, valid_df, test_df = load_data()
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_features(train_df, valid_df, test_df)
    
    y_train_min = y_train.min()
    
    # Load or train individual base models
    trained_models, base_results = load_or_train_base_models(X_train, y_train, X_valid, y_valid, base_models_dir='base_models')
    
    # Display menu and get user choice
    print("\n" + "=" * 70)
    print("ENSEMBLE METHOD SELECTION")
    print("=" * 70)
    print("Choose which ensemble method(s) to run:")
    print()
    print("  1. Hard Voting Ensemble")
    print("  2. Soft Voting Ensemble")
    print("  3. Weighted Voting Ensemble")
    print("  4. Stacking Ensemble (Random Forest meta-learner)")
    print("  5. Stacking Ensemble (Logistic Regression meta-learner)")
    print("  6. Run ALL ensemble methods")
    print()
    print("=" * 70)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice not in ['1', '2', '3', '4', '5', '6']:
        print("\n❌ Invalid choice! Running all methods by default...")
        choice = '6'
    
    # Determine which methods to run
    methods_to_run = []
    if choice == '1':
        methods_to_run = ['Hard Voting']
    elif choice == '2':
        methods_to_run = ['Soft Voting']
    elif choice == '3':
        methods_to_run = ['Weighted Voting']
    elif choice == '4':
        methods_to_run = ['Stacking (RF)']
    elif choice == '5':
        methods_to_run = ['Stacking (LR)']
    else:  # choice == '6'
        methods_to_run = ['Hard Voting', 'Soft Voting', 'Weighted Voting', 'Stacking (RF)', 'Stacking (LR)']
    
    # ═══════════════════════════════════════════════════════════════════
    # ENSEMBLE METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ENSEMBLE METHODS - VALIDATION SET")
    print("=" * 60)
    
    ensemble_results = {}
    ensemble_models = {}
    best_weights = None
    
    # 1. Hard Voting
    if 'Hard Voting' in methods_to_run:
        print("\n1. Hard Voting...")
        hard_voting = load_ensemble_model('hard_voting_ensemble.pkl')
        if hard_voting is None:
            hard_voting = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='hard'
            )
            hard_voting.fit(X_train, y_train - y_train_min)
            save_ensemble_model(hard_voting, 'hard_voting_ensemble.pkl')
        ensemble_models['Hard Voting'] = hard_voting
        hard_acc, hard_f1, hard_pred = hard_voting_ensemble(trained_models, X_valid, y_valid, y_train_min)
        ensemble_results['Hard Voting'] = {'accuracy': hard_acc, 'f1': hard_f1}
        print(f"   ✓ Hard Voting: Accuracy = {hard_acc:.4f}, Macro F1 = {hard_f1:.4f}")
    
    # 2. Soft Voting
    if 'Soft Voting' in methods_to_run:
        print("\n2. Soft Voting...")
        soft_voting = load_ensemble_model('soft_voting_ensemble.pkl')
        if soft_voting is None:
            soft_voting = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft'
            )
            soft_voting.fit(X_train, y_train - y_train_min)
            save_ensemble_model(soft_voting, 'soft_voting_ensemble.pkl')
        ensemble_models['Soft Voting'] = soft_voting
        soft_acc, soft_f1, soft_pred = soft_voting_ensemble(trained_models, X_valid, y_valid, y_train_min)
        ensemble_results['Soft Voting'] = {'accuracy': soft_acc, 'f1': soft_f1}
        print(f"   ✓ Soft Voting: Accuracy = {soft_acc:.4f}, Macro F1 = {soft_f1:.4f}")
    
    # 3. Weighted Voting (with optimization)
    if 'Weighted Voting' in methods_to_run:
        print("\n3. Weighted Voting...")
        weighted_voting = load_ensemble_model('weighted_voting_ensemble.pkl')
        if weighted_voting is None:
            print("   Searching for optimal weights...")
            best_weights, _ = find_best_weights(trained_models, X_valid, y_valid, y_train_min)
            weighted_voting = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft',
                weights=[best_weights['Random Forest'], best_weights['XGBoost'], best_weights['MLP']]
            )
            weighted_voting.fit(X_train, y_train - y_train_min)
            save_ensemble_model(weighted_voting, 'weighted_voting_ensemble.pkl')
        else:
            best_weights = {
                'Random Forest': weighted_voting.weights[0],
                'XGBoost': weighted_voting.weights[1],
                'MLP': weighted_voting.weights[2]
            }
        ensemble_models['Weighted Voting'] = weighted_voting
        weighted_acc, weighted_f1, weighted_pred, weights = weighted_voting_ensemble(
            trained_models, X_valid, y_valid, y_train_min, best_weights
        )
        ensemble_results['Weighted Voting'] = {'accuracy': weighted_acc, 'f1': weighted_f1, 'weights': weights}
        print(f"   Best weights: RF={weights['Random Forest']:.2f}, XGB={weights['XGBoost']:.2f}, MLP={weights['MLP']:.2f}")
        print(f"   ✓ Weighted Voting: Accuracy = {weighted_acc:.4f}, Macro F1 = {weighted_f1:.4f}")
    
    # 4. Stacking (with RF final estimator)
    if 'Stacking (RF)' in methods_to_run:
        print("\n4. Stacking (Random Forest)...")
        stacking_rf = load_ensemble_model('stacking_rf_ensemble.pkl')
        if stacking_rf is None:
            stacking_rf, stack_rf_acc, stack_rf_f1, stack_rf_name, stack_rf_pred = stacking_ensemble(
                X_train, y_train, X_valid, y_valid, trained_models, final_estimator='rf'
            )
            save_ensemble_model(stacking_rf, 'stacking_rf_ensemble.pkl')
        else:
            stack_rf_name = "Stacking (Random Forest)"
            stack_rf_pred = stacking_rf.predict(X_valid) + y_train_min
            stack_rf_acc = accuracy_score(y_valid, stack_rf_pred)
            stack_rf_f1 = f1_score(y_valid, stack_rf_pred, average='macro', zero_division=0)
        ensemble_results['Stacking (RF)'] = {'accuracy': stack_rf_acc, 'f1': stack_rf_f1}
        ensemble_models['Stacking (RF)'] = stacking_rf
        print(f"   ✓ {stack_rf_name}: Accuracy = {stack_rf_acc:.4f}, Macro F1 = {stack_rf_f1:.4f}")
    
    # 5. Stacking with Logistic Regression (NOVELTY)
    if 'Stacking (LR)' in methods_to_run:
        print("\n5. Stacking (Logistic Regression) - NOVELTY...")
        stacking_lr = load_ensemble_model('stacking_lr_ensemble.pkl')
        if stacking_lr is None:
            stacking_lr, stack_lr_acc, stack_lr_f1, stack_lr_name, stack_lr_pred = stacking_ensemble(
                X_train, y_train, X_valid, y_valid, trained_models, final_estimator='lr'
            )
            save_ensemble_model(stacking_lr, 'stacking_lr_ensemble.pkl')
        else:
            stack_lr_name = "Stacking (Logistic Regression)"
            stack_lr_pred = stacking_lr.predict(X_valid) + y_train_min
            stack_lr_acc = accuracy_score(y_valid, stack_lr_pred)
            stack_lr_f1 = f1_score(y_valid, stack_lr_pred, average='macro', zero_division=0)
        ensemble_results['Stacking (LR)'] = {'accuracy': stack_lr_acc, 'f1': stack_lr_f1}
        ensemble_models['Stacking (LR)'] = stacking_lr
        print(f"   ✓ {stack_lr_name}: Accuracy = {stack_lr_acc:.4f}, Macro F1 = {stack_lr_f1:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL TEST SET EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_results = {}
    
    # Individual models on test set (always show for reference)
    print("\n--- Base Models ---")
    for name, model in trained_models.items():
        if name == 'XGBoost':
            y_test_pred = model.predict(X_test) + y_train_min
        else:
            y_test_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_results[name] = {'accuracy': acc, 'f1': f1}
        print(f"  {name}: Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")
    
    # Ensemble methods on test set
    print("\n--- Ensemble Methods ---")
    
    # Hard Voting
    if 'Hard Voting' in methods_to_run:
        hard_test_acc, hard_test_f1, _ = hard_voting_ensemble(trained_models, X_test, y_test, y_train_min)
        test_results['Hard Voting'] = {'accuracy': hard_test_acc, 'f1': hard_test_f1}
        print(f"  Hard Voting: Accuracy = {hard_test_acc:.4f}, Macro F1 = {hard_test_f1:.4f}")
    
    # Soft Voting
    if 'Soft Voting' in methods_to_run:
        soft_test_acc, soft_test_f1, _ = soft_voting_ensemble(trained_models, X_test, y_test, y_train_min)
        test_results['Soft Voting'] = {'accuracy': soft_test_acc, 'f1': soft_test_f1}
        print(f"  Soft Voting: Accuracy = {soft_test_acc:.4f}, Macro F1 = {soft_test_f1:.4f}")
    
    # Weighted Voting
    if 'Weighted Voting' in methods_to_run:
        weighted_test_acc, weighted_test_f1, _, _ = weighted_voting_ensemble(
            trained_models, X_test, y_test, y_train_min, best_weights
        )
        test_results['Weighted Voting'] = {'accuracy': weighted_test_acc, 'f1': weighted_test_f1}
        print(f"  Weighted Voting: Accuracy = {weighted_test_acc:.4f}, Macro F1 = {weighted_test_f1:.4f}")
    
    # # Stacking (RF)
    if 'Stacking (RF)' in methods_to_run:
        stacking_rf = ensemble_models['Stacking (RF)']
        stack_rf_test_pred = stacking_rf.predict(X_test) + y_train_min
        stack_rf_test_acc = accuracy_score(y_test, stack_rf_test_pred)
        stack_rf_test_f1 = f1_score(y_test, stack_rf_test_pred, average='macro', zero_division=0)
        test_results['Stacking (RF)'] = {'accuracy': stack_rf_test_acc, 'f1': stack_rf_test_f1}
        print(f"  Stacking (RF): Accuracy = {stack_rf_test_acc:.4f}, Macro F1 = {stack_rf_test_f1:.4f}")
    
    # Stacking (LR) - NOVELTY
    if 'Stacking (LR)' in methods_to_run:
        stacking_lr = ensemble_models['Stacking (LR)']
        stack_lr_test_pred = stacking_lr.predict(X_test) + y_train_min
        stack_lr_test_acc = accuracy_score(y_test, stack_lr_test_pred)
        stack_lr_test_f1 = f1_score(y_test, stack_lr_test_pred, average='macro', zero_division=0)
        test_results['Stacking (LR)'] = {'accuracy': stack_lr_test_acc, 'f1': stack_lr_test_f1}
        print(f"  Stacking (LR) [NOVELTY]: Accuracy = {stack_lr_test_acc:.4f}, Macro F1 = {stack_lr_test_f1:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("SUMMARY - ACCURACY COMPARISON")
    print("=" * 70)
    
    summary_data = []
    
    # Base models
    for name in ['Random Forest', 'XGBoost', 'MLP']:
        summary_data.append({
            'Model': name,
            'Type': 'Base Model',
            'Validation Accuracy': f"{base_results[name]['accuracy']:.4f}",
            'Test Accuracy': f"{test_results[name]['accuracy']:.4f}",
            'Test Macro F1': f"{test_results[name]['f1']:.4f}"
        })
    
    # Ensemble methods (only those that were run)
    for name in methods_to_run:
        model_type = 'Ensemble'
        if name == 'Stacking (LR)':
            model_type = 'Ensemble (NOVELTY)'
        
        summary_data.append({
            'Model': name,
            'Type': model_type,
            'Validation Accuracy': f"{ensemble_results[name]['accuracy']:.4f}",
            'Test Accuracy': f"{test_results[name]['accuracy']:.4f}",
            'Test Macro F1': f"{test_results[name]['f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Find best model
    best_test = max(test_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_test[0]} with Test Accuracy = {best_test[1]['accuracy']:.4f}")
    
    # Save results
    output_file = f"ensemble_results_choice_{choice}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to '{output_file}'")


if __name__ == "__main__":
    main()