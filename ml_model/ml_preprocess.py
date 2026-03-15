import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


VALID_RANGES = {
    "temperature": (91.4, 107.6),
    "heartrate": (10, 300),
    "resprate": (3, 60),
    "o2sat": (60, 100),
    "sbp": (30, 300),
    "dbp": (30, 300),
    "pain": (0, 10),
}


# =====================
# Phase 1: Relabel Data
# =====================

def parse_pain(p):
    try:
        return float(p)
    except:
        return np.nan


def vitals_score(row):
    s = 0
    if pd.notna(row.get('heartrate')) and (row['heartrate'] > 100 or row['heartrate'] < 60):
        s += 1
    if pd.notna(row.get('resprate')) and (row['resprate'] > 20 or row['resprate'] < 8):
        s += 1
    if pd.notna(row.get('o2sat')) and row['o2sat'] < 90:
        s += 1
    if pd.notna(row.get('sbp')) and (row['sbp'] < 100 or row['sbp'] > 200):
        s += 1
    if pd.notna(row.get('temperature')) and (row['temperature'] > 100.4 or row['temperature'] < 96.8):
        s += 1
    return s


def relabel(row, cc_dist):
    cc = row['cc_clean']
    if cc not in cc_dist.index:
        return row['acuity']
    dist = cc_dist.loc[cc]
    majority_class = dist.idxmax()
    majority_pct = dist.max()

    # Rule 1: Clear complaints (majority >= 65%) -> relabel to majority
    if majority_pct >= 0.65:
        return majority_class

    # Rule 2: Ambiguous complaints (< 65%) -> use vital_score
    vs = row['vital_score']
    if vs >= 2:
        return 2
    elif vs == 0:
        return 3
    else:
        return row['acuity']  # vital_score == 1 -> keep original


def run_relabeling():
    relabel_dir = 'relabelled_data'
    os.makedirs(relabel_dir, exist_ok=True)

    # Load original data
    train = pd.read_csv('../raw_data/triage_train.csv').dropna(subset=['acuity'])
    test = pd.read_csv('../raw_data/triage_test.csv').dropna(subset=['acuity'])
    valid = pd.read_csv('../raw_data/triage_valid.csv').dropna(subset=['acuity'])

    # Merge classes 4,5 into 3
    for df in [train, test, valid]:
        df['acuity'] = df['acuity'].astype(int)
        df.loc[df['acuity'] >= 4, 'acuity'] = 3
        df['cc_clean'] = df['chiefcomplaint'].fillna('unknown').str.lower().str.strip()
        df['vital_score'] = df.apply(vitals_score, axis=1)

    # Build complaint distribution from TRAINING data only
    cc_dist = train.groupby('cc_clean')['acuity'].value_counts(normalize=True).unstack(fill_value=0)

    # Quick diagnostics to verify relabeling candidates exist
    majority_share = cc_dist.max(axis=1)
    clear_complaints = int((majority_share >= 0.65).sum())
    ambiguous_complaints = int((majority_share < 0.65).sum())
    print(f"complaints with clear majority (>=65%): {clear_complaints}")
    print(f"complaints ambiguous (<65%): {ambiguous_complaints}")

    # Apply relabeling to all splits using training-derived rules
    for name, df in [('train', train), ('test', test), ('valid', valid)]:
        original = df['acuity'].copy()
        df['acuity'] = df.apply(lambda r: relabel(r, cc_dist), axis=1)
        changed = (original != df['acuity']).sum()
        print(f"{name}: {changed}/{len(df)} rows relabeled ({changed/len(df)*100:.1f}%)")
        out_file = os.path.join(relabel_dir, f'triage_{name}_relabeled.csv')
        df.drop(columns=['cc_clean', 'vital_score']).to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

    print("\nSaved: relabelled_data/triage_train_relabeled.csv, relabelled_data/triage_test_relabeled.csv, relabelled_data/triage_valid_relabeled.csv")


# =======================
# Phase 2: Preprocess Data
# =======================

def clean_text(text):
    if pd.isna(text) or text == '':
        return 'unknown'
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text if text else 'unknown'


def preprocess_file(input_file, scaler=None):
    print(f"\nProcessing {input_file}...")
    df = pd.read_csv(input_file)
    original_len = len(df)

    df['chiefcomplaint_clean'] = df['chiefcomplaint'].apply(clean_text)
    df.drop(columns=['subject_id', 'stay_id', 'chiefcomplaint'], inplace=True, errors='ignore')

    dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"  Removed {dupes} duplicates")

    df['pain'] = pd.to_numeric(df['pain'], errors='coerce')

    numeric_cols = [c for c in df.columns if c != 'chiefcomplaint_clean']
    missing = df[numeric_cols].isnull().any(axis=1).sum()
    df = df[~df[numeric_cols].isnull().any(axis=1)]
    print(f"  Removed {missing} rows with missing values")

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    exclude = ['acuity', 'chiefcomplaint_clean']
    cols_to_scale = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    print(f"  {original_len} -> {len(df)} rows, {df.shape[1]} cols")
    return df, scaler


def run_preprocessing():
    output_dir = 'ml_processed_data'
    os.makedirs(output_dir, exist_ok=True)

    relabel_dir = 'relabelled_data'

    train_df, fitted_scaler = preprocess_file(os.path.join(relabel_dir, 'triage_train_relabeled.csv'))
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(fitted_scaler, f)
    print(f"Saved: {scaler_path}")

    valid_df, _ = preprocess_file(os.path.join(relabel_dir, 'triage_valid_relabeled.csv'), scaler=fitted_scaler)
    test_df, _ = preprocess_file(os.path.join(relabel_dir, 'triage_test_relabeled.csv'), scaler=fitted_scaler)

    print("\nFitting TF-IDF on training data...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, stop_words='english')
    vectorizer.fit(train_df['chiefcomplaint_clean'])
    tfidf_vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(tfidf_vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved: {tfidf_vectorizer_path}")

    for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
        tfidf = vectorizer.transform(df['chiefcomplaint_clean'])
        feat_names = [f'tfidf_{n}' for n in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=feat_names, index=df.index)
        final = pd.concat([df.drop(columns=['chiefcomplaint_clean']), tfidf_df], axis=1)
        out_path = os.path.join(output_dir, f'ml_processed_{name}.csv')
        final.to_csv(out_path, index=False)
        print(f"  {name}: {final.shape}")
        print(f"Saved: {out_path}")

    feature_order = final.drop(columns=['acuity']).columns.tolist()
    feature_order_path = os.path.join(output_dir, 'feature_order.pkl')
    with open(feature_order_path, 'wb') as f:
        pickle.dump(feature_order, f)
    print(f"Saved: {feature_order_path}")

    print(f"\nDone! {len(feature_order)} features saved to {output_dir}/")


if __name__ == '__main__':
    run_relabeling()
    run_preprocessing()