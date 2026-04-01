import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, f1_score
from lightgbm import LGBMClassifier

# ============================================================
# CONFIG
# ============================================================

VALID_RANGES = {
    "temperature": (91.4, 107.6), "heartrate": (10, 300), "resprate": (3, 60),
    "o2sat": (60, 100), "sbp": (30, 300), "dbp": (30, 300), "pain": (0, 10),
}

VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]

# 64 complaint groups + other (from our analysis — covers 95% of patients)
COMPLAINT_GROUPS = {
    'abdominal pain': ['abdominal pain', 'abdo pain'],
    'shortness of breath': ['shortness of breath'],
    'fall': ['status post fall', 'fall'],
    'chest pain': ['chest pain'],
    'nausea vomiting': ['nausea vomiting', 'nausea', 'vomiting'],
    'weakness': ['weakness'],
    'altered mental status': ['altered mental status', 'altered ms', 'ms changes',
                              'mental status change', 'altered level of consciousness'],
    'transfer': ['transfer'],
    'fever': ['fever'],
    'abnormal labs': ['abnormal labs', 'abnormal lab'],
    'back pain': ['back pain', 'lower back pain'],
    'wound evaluation': ['wound eval'],
    'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'vertigo'],
    'swelling': ['swelling', 'edema'],
    'headache': ['headache', 'migraine'],
    'rectal bleeding': ['bright red blood per rectum', 'rectal bleeding', 'blood in stool', 'rectal pain'],
    'pain general': ['pain'],
    'motor vehicle collision': ['motor vehicle collision', 'motor vehicle accident',
                                'motorcycle accident', 'mcc'],
    'intracranial hemorrhage': ['intracranial hemorrhage', 'sah', 'sdh', 'head bleed'],
    'syncope': ['syncope', 'presyncope'],
    'seizure': ['seizure'],
    'fracture': ['fracture'],
    'leg pain': ['leg pain', 'lower extremity pain'],
    'stroke': ['stroke', 'cerebrovascular accident', 'code stroke'],
    'confusion': ['confusion', 'agitation'],
    'hypotension': ['hypotension'],
    'gi bleed': ['gastrointestinal bleed', 'melena', 'hematemesis', 'coffee ground'],
    'abnormal imaging': ['abnormal ct', 'abnormal mri'],
    'stroke symptoms': ['slurred speech', 'facial droop', 'aphasia'],
    'cough': ['cough'],
    'flank pain': ['flank pain'],
    'abdominal distention': ['abdominal distention'],
    'numbness': ['numbness', 'tingling'],
    'foot pain': ['foot pain', 'toe pain'],
    'vision changes': ['visual changes', 'vision changes', 'blurred vision'],
    'tachycardia': ['tachycardia'],
    'injury': ['injury'],
    'cardiac': ['cardiac', 'stemi', 'nstemi', 'atrial fibrillation', 'abnormal ekg'],
    'hip pain': ['hip pain'],
    'hyperglycemia': ['hyperglycemia'],
    'epigastric pain': ['epigastric pain'],
    'diarrhea': ['diarrhea'],
    'gait mobility issues': ['unsteady gait', 'unable to ambulate'],
    'palpitations': ['palpitations'],
    'unresponsive': ['unresponsive', 'found down'],
    'respiratory distress': ['respiratory distress', 'resp distress'],
    'lethargy': ['lethargy'],
    'knee pain': ['knee pain'],
    'cellulitis': ['cellulitis'],
    'electrolyte abnormality': ['hyperkalemia', 'hypokalemia', 'hyponatremia', 'abnormal sodium'],
    'neck pain': ['neck pain'],
    'failure to thrive': ['failure to thrive', 'ftt'],
    'hematuria': ['hematuria'],
    'pedestrian struck': ['pedestrian struck'],
    'hypertension': ['hypertension'],
    'hypoxia': ['hypoxia'],
    'bradycardia': ['bradycardia'],
    'flu ili': ['influenza', 'ili'],
    'urinary issues': ['urinary retention'],
    'pneumonia': ['pneumonia'],
    'suicidal ideation': ['suicidal ideation'],
    'anemia': ['anemia'],
    'arm pain': ['arm pain'],
}

GROUP_COLUMNS = list(COMPLAINT_GROUPS.keys()) + ['other']


# ============================================================
# PHASE 1: RELABELING (from ml_preprocess.py logic)
# ============================================================

def vitals_score(row):
    s = 0
    if pd.notna(row.get('heartrate')) and (row['heartrate'] > 100 or row['heartrate'] < 60): s += 1
    if pd.notna(row.get('resprate')) and (row['resprate'] > 20 or row['resprate'] < 8): s += 1
    if pd.notna(row.get('o2sat')) and row['o2sat'] < 90: s += 1
    if pd.notna(row.get('sbp')) and (row['sbp'] < 100 or row['sbp'] > 200): s += 1
    if pd.notna(row.get('temperature')) and (row['temperature'] > 100.4 or row['temperature'] < 96.8): s += 1
    return s


def relabel(row, cc_dist):
    cc = row['cc_clean']
    if cc not in cc_dist.index:
        return row['acuity']
    dist = cc_dist.loc[cc]
    majority_class = dist.idxmax()
    majority_pct = dist.max()
    if majority_pct >= 0.65:
        return majority_class
    vs = row['vital_score']
    if vs >= 2: return 2
    elif vs == 0: return 3
    else: return row['acuity']


def run_relabeling():
    print("=" * 60)
    print("PHASE 1: Relabeling")
    print("=" * 60)

    # Load from temp_data (normalized complaints)
    train = pd.read_csv('../temp_data/triage_train_relabeled.csv').dropna(subset=['acuity'])
    test = pd.read_csv('../temp_data/triage_test_relabeled.csv').dropna(subset=['acuity'])
    valid = pd.read_csv('../temp_data/triage_valid_relabeled.csv').dropna(subset=['acuity'])

    for df in [train, test, valid]:
        df['acuity'] = df['acuity'].astype(int)
        df.loc[df['acuity'] >= 4, 'acuity'] = 3
        df['cc_clean'] = df['chiefcomplaint'].fillna('unknown').str.lower().str.strip()
        df['vital_score'] = df.apply(vitals_score, axis=1)

    # Build complaint distribution from TRAINING data only
    cc_dist = train.groupby('cc_clean')['acuity'].value_counts(normalize=True).unstack(fill_value=0)

    for name, df in [('train', train), ('test', test), ('valid', valid)]:
        original = df['acuity'].copy()
        df['acuity'] = df.apply(lambda r: relabel(r, cc_dist), axis=1)
        changed = (original != df['acuity']).sum()
        print(f"  {name}: {changed}/{len(df)} relabeled ({changed / len(df) * 100:.1f}%)")

    # Drop helper columns
    for df in [train, test, valid]:
        df.drop(columns=['cc_clean', 'vital_score'], inplace=True)

    return train, valid, test


# ============================================================
# PHASE 2: BINARY COMPLAINT ENCODING + VITALS PREPROCESSING
# ============================================================

def encode_complaints(text):
    """Multi-hot encode a complaint string into 65 binary features."""
    result = {g: 0 for g in GROUP_COLUMNS}
    if pd.isna(text):
        result['other'] = 1
        return result

    t = text.lower().strip()
    matched = False

    # Check each group — longest keywords first to avoid partial matches
    for group, keywords in COMPLAINT_GROUPS.items():
        for kw in keywords:
            if kw in t:
                result[group] = 1
                matched = True
                break

    if not matched:
        result['other'] = 1

    return result


def preprocess(df, scaler=None, fit_scaler=False):
    """Preprocess a dataframe: encode complaints, clean vitals, scale."""
    original_len = len(df)

    # Drop IDs
    df = df.drop(columns=['subject_id', 'stay_id'], errors='ignore').copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Parse pain as numeric
    df['pain'] = pd.to_numeric(df['pain'], errors='coerce')

    # Drop rows with missing vitals
    df = df.dropna(subset=VITALS + ['acuity'])

    # Clip vitals to valid ranges
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Encode complaints into binary columns
    complaint_features = df['chiefcomplaint'].apply(encode_complaints).apply(pd.Series)
    complaint_features.index = df.index

    # Scale vitals
    if fit_scaler:
        scaler = StandardScaler()
        df[VITALS] = scaler.fit_transform(df[VITALS])
    else:
        df[VITALS] = scaler.transform(df[VITALS])

    # Combine: vitals + complaint binary features
    final = pd.concat([df[VITALS], complaint_features, df[['acuity']]], axis=1)

    print(f"  {original_len} -> {len(final)} rows, {final.shape[1]} cols "
          f"({len(VITALS)} vitals + {len(GROUP_COLUMNS)} complaint groups + acuity)")

    return final, scaler


def run_preprocessing(train, valid, test):
    print("\n" + "=" * 60)
    print("PHASE 2: Preprocessing (binary complaint encoding + vitals)")
    print("=" * 60)

    train_proc, scaler = preprocess(train, fit_scaler=True)
    valid_proc, _ = preprocess(valid, scaler=scaler)
    test_proc, _ = preprocess(test, scaler=scaler)

    # Save scaler
    os.makedirs('processed_data', exist_ok=True)
    with open('processed_data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('processed_data/complaint_groups.pkl', 'wb') as f:
        pickle.dump(COMPLAINT_GROUPS, f)

    return train_proc, valid_proc, test_proc


# ============================================================
# PHASE 3: TRAIN LIGHTGBM (DEFAULT)
# ============================================================

def train_and_evaluate(train_proc, valid_proc, test_proc):
    print("\n" + "=" * 60)
    print("PHASE 3: Training LightGBM (default parameters)")
    print("=" * 60)

    feature_cols = [c for c in train_proc.columns if c != 'acuity']
    X_train = train_proc[feature_cols]
    y_train = train_proc['acuity'].astype(int)
    X_valid = valid_proc[feature_cols]
    y_valid = valid_proc['acuity'].astype(int)
    X_test = test_proc[feature_cols]
    y_test = test_proc['acuity'].astype(int)

    print(f"  Features: {len(feature_cols)} ({len(VITALS)} vitals + {len(GROUP_COLUMNS)} complaint groups)")
    print(f"  Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Class distribution (train): {y_train.value_counts().sort_index().to_dict()}")

    # LightGBM expects 0-indexed labels
    y_train_lgb = y_train - 1
    classes = np.unique(y_train_lgb)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_lgb)
    sample_weights = np.array([class_weights[int(y)] for y in y_train_lgb])

    model = LGBMClassifier()
    model.fit(X_train, y_train_lgb, sample_weight=sample_weights)

    # Save model
    with open('processed_data/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate
    for name, X, y in [('Validation', X_valid, y_valid), ('Test', X_test, y_test)]:
        y_pred = model.predict(X) + 1
        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        print(f"\n{'=' * 50}")
        print(f"{name} Results:")
        print(f"{'=' * 50}")
        print(f"Accuracy:         {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Macro F1:         {f1_macro:.4f}")
        print(f"Weighted F1:      {f1_weighted:.4f}")
        print(f"\n{classification_report(y, y_pred, target_names=['Critical(1)', 'Urgent(2)', 'Non-Urgent(3)'])}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 20 most important features:")
    for feat, imp in importance.head(20).items():
        print(f"  {feat:<35} {imp}")

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    train, valid, test = run_relabeling()
    train_proc, valid_proc, test_proc = run_preprocessing(train, valid, test)
    model = train_and_evaluate(train_proc, valid_proc, test_proc)
    print("\nDone! Model and artifacts saved to processed_data/")
