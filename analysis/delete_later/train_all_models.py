import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('ml_processed_data/ml_processed_train.csv')
test = pd.read_csv('ml_processed_data/ml_processed_test.csv')

X_tr = train.drop(columns=['acuity']).values
y_tr = train['acuity'].values
X_te = test.drop(columns=['acuity']).values
y_te = test['acuity'].values

print(f"Train: {X_tr.shape}, Test: {X_te.shape}\n")

# All default parameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'LightGBM': lgb.LGBMClassifier(verbose=-1),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', verbosity=0),
    'CatBoost': CatBoostClassifier(verbose=0),
    'Linear SVM': LinearSVC(max_iter=2000),
    'SVC (RBF)': SVC(),
    'MLP': MLPClassifier(max_iter=500),
}

classes = [1, 2, 3]
rows = []
for name, model in models.items():
    print(f"Training {name}...", end=' ', flush=True)
    if name in ('LightGBM', 'XGBoost'):
        model.fit(X_tr, y_tr - 1)
        y_pred = model.predict(X_te) + 1
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, labels=classes)
    pw, rw, fw, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted')

    row = {'Model': name, 'Accuracy': round(acc, 4)}
    for i, c in enumerate(classes):
        row[f'P_class{c}'] = round(p[i], 4)
        row[f'R_class{c}'] = round(r[i], 4)
        row[f'F1_class{c}'] = round(f[i], 4)
    row['P_weighted'] = round(pw, 4)
    row['R_weighted'] = round(rw, 4)
    row['F1_weighted'] = round(fw, 4)
    rows.append(row)
    print(f"{acc:.4f}")

res_df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False).reset_index(drop=True)
res_df.index = range(1, len(res_df) + 1)
res_df.index.name = 'Rank'

# Print summary
print("\n" + "="*120)
print(f"{'Rank':<5}{'Model':<22}{'Acc':>7} | {'P1':>6}{'R1':>6}{'F1_1':>6} | {'P2':>6}{'R2':>6}{'F1_2':>6} | {'P3':>6}{'R3':>6}{'F1_3':>6} | {'Pw':>6}{'Rw':>6}{'F1w':>6}")
print("="*120)
for i, r in res_df.iterrows():
    print(f"{i:<5}{r['Model']:<22}{r['Accuracy']:>6.2%} | "
          f"{r['P_class1']:>5.2%}{r['R_class1']:>6.2%}{r['F1_class1']:>6.2%} | "
          f"{r['P_class2']:>5.2%}{r['R_class2']:>6.2%}{r['F1_class2']:>6.2%} | "
          f"{r['P_class3']:>5.2%}{r['R_class3']:>6.2%}{r['F1_class3']:>6.2%} | "
          f"{r['P_weighted']:>5.2%}{r['R_weighted']:>6.2%}{r['F1_weighted']:>6.2%}")
print("="*120)

res_df.to_csv('ml_processed_data/model_results.csv')
print("\nResults saved to ml_processed_data/model_results.csv")
