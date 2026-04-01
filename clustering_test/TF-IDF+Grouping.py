"""
Train all models using combined features: 7 vitals + 97 binary complaint groups + 500 TF-IDF.

Pipeline:
  1. Relabel using complaint majority class + vital scores (from temp_data)
  2. Preprocess: scale vitals, encode binary groups, fit TF-IDF
  3. Train 7 models with default parameters and evaluate

Input:  ../temp_data/triage_{train,valid,test}_relabeled.csv
Output: Console results comparing all models
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

VALID_RANGES = {
    "temperature": (91.4, 107.6), "heartrate": (10, 300), "resprate": (3, 60),
    "o2sat": (60, 100), "sbp": (30, 300), "dbp": (30, 300), "pain": (0, 10),
}
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]

COMPLAINT_GROUPS = {
    'abdominal distention': ['abdominal distention'],
    'abdominal pain': ['abdominal pain', 'abdo pain'],
    'abnormal imaging': ['abnormal ct', 'abnormal mri'],
    'abnormal labs': ['abnormal labs', 'abnormal lab'],
    'abscess': ['abscess'], 'alcohol intoxication': ['alcohol'],
    'allergic reaction': ['allergic reaction'],
    'altered mental status': ['altered mental status', 'altered ms', 'ms changes',
                              'mental status change', 'altered level of consciousness'],
    'anemia': ['anemia'], 'anxiety': ['anxiety'], 'arm pain': ['arm pain'],
    'assault': ['assault'], 'asthma': ['asthma'],
    'back pain': ['back pain', 'lower back pain'],
    'bicycle accident': ['bicycle accident', 'bicycle crash'],
    'body pain': ['body pain'], 'bowel obstruction': ['sbo'],
    'bradycardia': ['bradycardia'],
    'cardiac': ['abnormal ekg', 'stemi', 'nstemi', 'atrial fibrillation'],
    'cardiac arrest': ['cardiac arrest'], 'cellulitis': ['cellulitis'],
    'chest pain': ['chest pain'], 'confusion': ['confusion', 'agitation'],
    'constipation': ['constipation'], 'cough': ['cough'],
    'deep vein thrombosis': ['deep vein thrombosis'], 'dehydration': ['dehydration'],
    'dialysis access issues': ['fistula'], 'diarrhea': ['diarrhea'],
    'difficulty swallowing': ['difficulty swallowing', 'dysphagia'],
    'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'vertigo'],
    'dysuria': ['dysuria'],
    'electrolyte abnormality': ['hyperkalemia', 'hypokalemia', 'hyponatremia', 'abnormal sodium'],
    'epigastric pain': ['epigastric pain'], 'evaluation': ['for eval'],
    'failure to thrive': ['failure to thrive', 'ftt'],
    'fall': ['status post fall', 'fall'], 'fatigue': ['fatigue'], 'fever': ['fever'],
    'flank pain': ['flank pain'], 'flu ili': ['influenza', ' ili'],
    'foot pain': ['foot pain', 'toe pain'], 'fracture': ['fracture'],
    'gait mobility issues': ['unsteady gait', 'unable to ambulate'],
    'gi bleed': ['gastrointestinal bleed', 'hematemesis', 'melena', 'coffee ground'],
    'head injury': ['head injury'], 'headache': ['headache', 'migraine'],
    'hematuria': ['hematuria'], 'hemoptysis': ['hemoptysis'], 'hip pain': ['hip pain'],
    'hyperglycemia': ['hyperglycemia'], 'hypertension': ['hypertension'],
    'hypoglycemia': ['hypoglycemia'], 'hypotension': ['hypotension'],
    'hypoxia': ['hypoxia'], 'injury': ['injury'],
    'intracranial hemorrhage': ['intracranial hemorrhage', 'sah', 'sdh', 'head bleed'],
    'jaundice': ['jaundice'], 'knee pain': ['knee pain'], 'laceration': ['laceration'],
    'leg pain': ['leg pain', 'lower extremity pain'], 'lethargy': ['lethargy'],
    'motor vehicle collision': ['motor vehicle collision', 'motor vehicle accident',
                                'motorcycle accident', 'mcc'],
    'nausea vomiting': ['nausea', 'vomiting'], 'neck pain': ['neck pain'],
    'neurological evaluation': ['neuro eval'], 'numbness': ['numbness', 'tingling'],
    'overdose': ['overdose'], 'pain general': ['pain'], 'palpitations': ['palpitations'],
    'pedestrian struck': ['pedestrian struck'], 'pelvic pain': ['pelvic pain'],
    'pneumonia': ['pneumonia'], 'pulmonary embolism': ['pulmonary embolism'],
    'rash': ['rash'],
    'rectal bleeding': ['bright red blood per rectum', 'rectal bleeding', 'rectal pain', 'blood in stool'],
    'respiratory distress': ['respiratory distress', 'resp distress'],
    'seizure': ['seizure'], 'sepsis': ['sepsis', 'positive blood cultures', 'urosepsis'],
    'shortness of breath': ['shortness of breath'], 'sore throat': ['sore throat'],
    'stroke': ['cerebrovascular accident', 'stroke', 'code stroke'],
    'stroke symptoms': ['slurred speech', 'facial droop', 'aphasia'],
    'suicidal ideation': ['suicidal ideation'], 'swelling': ['swelling', 'edema'],
    'syncope': ['syncope', 'presyncope'], 'tachycardia': ['tachycardia'],
    'transfer': ['transfer'],
    'tube device evaluation': ['tube eval', 'gtube eval', 'jtube eval', 'nephrostomy'],
    'unresponsive': ['unresponsive', 'found down'], 'urinary issues': ['urinary retention'],
    'urinary tract infection': ['urinary tract infection'],
    'vaginal bleeding': ['vaginal bleeding'],
    'vision changes': ['visual changes', 'vision changes', 'blurred vision'],
    'weakness': ['weakness'], 'wound evaluation': ['wound eval'],
}
GROUP_COLUMNS = list(COMPLAINT_GROUPS.keys()) + ['other']


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
    if cc not in cc_dist.index: return row['acuity']
    dist = cc_dist.loc[cc]
    if dist.max() >= 0.75: return dist.idxmax()
    vs = row['vital_score']
    if vs >= 2: return 2
    elif vs == 0: return 3
    return row['acuity']


def encode_complaints(text):
    result = {g: 0 for g in GROUP_COLUMNS}
    if pd.isna(text):
        result['other'] = 1
        return result
    t = text.lower().strip()
    matched = False
    for group, keywords in COMPLAINT_GROUPS.items():
        for kw in keywords:
            if kw in t:
                result[group] = 1; matched = True; break
    if not matched: result['other'] = 1
    return result


def clean_text(text):
    if pd.isna(text) or text == '': return 'unknown'
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text if text else 'unknown'


def main():
    # Phase 1: Relabel
    print("Phase 1: Relabeling...")
    train = pd.read_csv('../temp_data/triage_train_relabeled.csv').dropna(subset=['acuity'])
    test = pd.read_csv('../temp_data/triage_test_relabeled.csv').dropna(subset=['acuity'])
    valid = pd.read_csv('../temp_data/triage_valid_relabeled.csv').dropna(subset=['acuity'])
    for df in [train, test, valid]:
        df['acuity'] = df['acuity'].astype(int)
        df.loc[df['acuity'] >= 4, 'acuity'] = 3
        df['cc_clean'] = df['chiefcomplaint'].fillna('unknown').str.lower().str.strip()
        df['vital_score'] = df.apply(vitals_score, axis=1)
    cc_dist = train.groupby('cc_clean')['acuity'].value_counts(normalize=True).unstack(fill_value=0)
    for name, df in [('train', train), ('test', test), ('valid', valid)]:
        orig = df['acuity'].copy()
        df['acuity'] = df.apply(lambda r: relabel(r, cc_dist), axis=1)
        print(f"  {name}: {(orig != df['acuity']).sum()}/{len(df)} relabeled")

    # Phase 2: Preprocess
    print(f"\nPhase 2: Preprocessing (vitals + {len(GROUP_COLUMNS)} groups + TF-IDF)...")
    scaler = None
    vectorizer = None
    datasets = {}
    for name, df in [('train', train), ('valid', valid), ('test', test)]:
        d = df.drop(columns=['subject_id', 'stay_id', 'cc_clean', 'vital_score'], errors='ignore').copy()
        d.drop_duplicates(inplace=True)
        d['pain'] = pd.to_numeric(d['pain'], errors='coerce')
        d = d.dropna(subset=VITALS + ['acuity'])
        for col, (lo, hi) in VALID_RANGES.items(): d[col] = d[col].clip(lower=lo, upper=hi)

        cf = d['chiefcomplaint'].apply(encode_complaints).apply(pd.Series)
        cf.index = d.index

        d['cc_text'] = d['chiefcomplaint'].apply(clean_text)
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=2, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(d['cc_text'])
        else:
            tfidf_matrix = vectorizer.transform(d['cc_text'])
        tfidf_cols = [f'tfidf_{n}' for n in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=d.index)

        if scaler is None:
            scaler = StandardScaler()
            d[VITALS] = scaler.fit_transform(d[VITALS])
        else:
            d[VITALS] = scaler.transform(d[VITALS])

        final = pd.concat([d[VITALS], cf, tfidf_df, d[['acuity']]], axis=1)
        datasets[name] = final
        print(f"  {name}: {final.shape}")

    feat = [c for c in datasets['train'].columns if c != 'acuity']
    X_tr, y_tr = datasets['train'][feat], datasets['train']['acuity'].astype(int)
    X_val, y_val = datasets['valid'][feat], datasets['valid']['acuity'].astype(int)
    X_te, y_te = datasets['test'][feat], datasets['test']['acuity'].astype(int)
    y_tr_0 = y_tr - 1
    cw = compute_class_weight('balanced', classes=np.unique(y_tr_0), y=y_tr_0)
    sw = np.array([cw[int(y)] for y in y_tr_0])

    print(f"\nFeatures: {len(feat)} (7 vitals + {len(GROUP_COLUMNS)} groups + {len(tfidf_cols)} tfidf)")

    # Phase 3: Train all models
    print("\nPhase 3: Training all models (default params)...")
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000), True, False),
        ('Random Forest',       RandomForestClassifier(random_state=42), True, False),
        ('MLP',                 MLPClassifier(max_iter=500, random_state=42), False, False),
        ('AdaBoost',            AdaBoostClassifier(random_state=42, algorithm='SAMME'), True, False),
        ('XGBoost',             XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0), True, True),
        ('LightGBM',            LGBMClassifier(verbose=-1), True, True),
        ('CatBoost',            CatBoostClassifier(random_state=42, verbose=0), True, True),
    ]

    results = []
    for name, model, use_sw, zero_idx in models:
        print(f"Training {name}...", end=" ", flush=True)
        y_fit = y_tr_0 if zero_idx else y_tr
        if use_sw:
            model.fit(X_tr, y_fit, sample_weight=sw)
        else:
            model.fit(X_tr, y_fit)
        offset = 1 if zero_idx else 0
        val_pred = model.predict(X_val) + offset
        test_pred = model.predict(X_te) + offset
        va = accuracy_score(y_val, val_pred)
        ta = accuracy_score(y_te, test_pred)
        tf1 = f1_score(y_te, test_pred, average='macro')
        wf1 = f1_score(y_te, test_pred, average='weighted')
        results.append({'Model': name, 'Val Acc': va, 'Test Acc': ta, 'Macro F1': tf1, 'Weighted F1': wf1})
        print(f"Val={va:.2%}, Test={ta:.2%}, Macro-F1={tf1:.4f}")

    print("\n" + "=" * 90)
    print(f"ALL MODELS — TF-IDF + BINARY GROUPS + VITALS = {len(feat)} FEATURES")
    print("=" * 90)
    for r in sorted(results, key=lambda x: -x['Test Acc']):
        print(f"  {r['Model']:<22} Val={r['Val Acc']:.2%}  Test={r['Test Acc']:.2%}  "
              f"Macro-F1={r['Macro F1']:.4f}  W-F1={r['Weighted F1']:.4f}")


if __name__ == '__main__':
    main()
