import pandas as pd
import os
import glob
import sys
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Define reasonable ranges for physiological variables
# Values outside these ranges are considered "impossible" or extreme outliers
VALID_RANGES = {
    "temperature": (91.4, 107.6),
    "heartrate": (10, 300),
    "resprate": (3, 60),
    "o2sat": (60, 100),
    "sbp": (30, 300),
    "dbp": (30, 300),
    "pain": (0, 10)
}

def clean_text(text):
    """
    Clean and normalize text data from chief complaint column.
    Steps:
    - Convert to lowercase
    - Remove special characters and numbers
    - Remove extra whitespace
    - Handle missing values
    """
    if pd.isna(text) or text == '':
        return 'unknown'
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text if text else 'unknown'

def handle_impossible_values(df):
    """
    Clip values outside of VALID_RANGES to valid range limits.
    """
    df_clean = df.copy()
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
    
    print("Clipped values to defined ranges.")
    print(f"Rows remaining: {len(df_clean)}")
    
    return df_clean

def scale_data(df, exclude_cols=['acuity'], scaler=None):
    """
    Scale numerical columns using Z-score normalization (StandardScaler).
    Ignores columns in exclude_cols.
    Returns both scaled dataframe and the scaler.
    """
    df_scaled = df.copy()
    
    # Identify columns to scale
    cols_to_scale = [col for col in df_scaled.columns 
                     if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_scaled[col])]
    
    if cols_to_scale:
        if scaler is None:
            # Fit new scaler (for training data)
            scaler = StandardScaler()
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            print("Scaled data (Z-score normalization) using StandardScaler.")
        else:
            # Use existing scaler (for validation/test data)
            df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])
            print("Scaled data using existing scaler.")
        
    return df_scaled, scaler

def preprocess_file(input_file, output_file, scaler=None):
    print(f"Processing {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return None, None

    df = pd.read_csv(input_file)
    original_len = len(df)
    
    # 1. Preprocess chief complaint column (text data)
    if 'chiefcomplaint' in df.columns:
        print(f"Preprocessing chief complaint column...")
        df['chiefcomplaint_clean'] = df['chiefcomplaint'].apply(clean_text)
        print(f"Sample cleaned chief complaints: {df['chiefcomplaint_clean'].head(3).tolist()}")
    
    # 2. Remove stay_id, subject_id (keep processed chiefcomplaint)
    cols_to_drop = ['subject_id', 'stay_id', 'chiefcomplaint']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"Dropped columns: {existing_cols_to_drop}")
    
    # 3. Remove duplicate rows
    # Note: We drop duplicates AFTER removing IDs, because rows might be identical except for ID
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicates_count} duplicate rows.")
    
    # Ensure columns are numeric (except text columns)
    for col in VALID_RANGES.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Remove missing rows from numeric columns only
    numeric_cols = [c for c in df.columns if c != 'chiefcomplaint_clean']
    missing_rows_count = df[numeric_cols].isnull().any(axis=1).sum()
    df = df[~df[numeric_cols].isnull().any(axis=1)]
    print(f"Removed {missing_rows_count} rows with missing numeric values.")
    
    # 5. Handle impossible values
    df = handle_impossible_values(df)
    
    # 6. Scale numeric data (exclude acuity and text)
    df, scaler = scale_data(df, exclude_cols=['acuity', 'chiefcomplaint_clean'], scaler=scaler)
    
    print(f"Shape before text vectorization: {df.shape} (Original rows: {original_len})")
    
    # Return dataframe and scaler
    return df, scaler

def vectorize_text_and_save(train_df, valid_df, test_df, output_dir):
    """
    Vectorize chief complaint text using TF-IDF and save all datasets.
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical features.
    """
    print("\n" + "="*60)
    print("TF-IDF Vectorization of Chief Complaint")
    print("="*60)
    
    # Check if chiefcomplaint_clean column exists
    if 'chiefcomplaint_clean' not in train_df.columns:
        print("No chief complaint column found. Saving without text features.")
        train_df.to_csv(os.path.join(output_dir, "ml_processed_train.csv"), index=False)
        valid_df.to_csv(os.path.join(output_dir, "ml_processed_valid.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "ml_processed_test.csv"), index=False)
        return
    
    # Initialize or load TF-IDF vectorizer (fit on train once, reuse thereafter)
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        print(f"Loading existing TF-IDF vectorizer from {vectorizer_path}...")
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        # Fit only on training data to avoid leakage, then persist for reuse
        print("Fitting TF-IDF vectorizer on training data and saving for reuse...")
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        vectorizer.fit(train_df['chiefcomplaint_clean'])
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Saved TF-IDF vectorizer to {vectorizer_path}")
    
    # Transform train/valid/test using the (saved) vectorizer
    train_text_features = vectorizer.transform(train_df['chiefcomplaint_clean'])
    valid_text_features = vectorizer.transform(valid_df['chiefcomplaint_clean'])
    test_text_features = vectorizer.transform(test_df['chiefcomplaint_clean'])
    
    print(f"TF-IDF features created: {train_text_features.shape[1]}")
    print(f"Top features: {vectorizer.get_feature_names_out()[:10].tolist()}")
    
    # Convert sparse matrix to dataframe
    feature_names = [f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
    
    train_text_df = pd.DataFrame(train_text_features.toarray(), columns=feature_names, index=train_df.index)
    valid_text_df = pd.DataFrame(valid_text_features.toarray(), columns=feature_names, index=valid_df.index)
    test_text_df = pd.DataFrame(test_text_features.toarray(), columns=feature_names, index=test_df.index)
    
    # Drop original text column and concatenate TF-IDF features
    train_final = pd.concat([train_df.drop(columns=['chiefcomplaint_clean']), train_text_df], axis=1)
    valid_final = pd.concat([valid_df.drop(columns=['chiefcomplaint_clean']), valid_text_df], axis=1)
    test_final = pd.concat([test_df.drop(columns=['chiefcomplaint_clean']), test_text_df], axis=1)
    
    print(f"\nFinal shapes:")
    print(f"Train: {train_final.shape}")
    print(f"Valid: {valid_final.shape}")
    print(f"Test: {test_final.shape}")
    
    # Save processed datasets
    train_final.to_csv(os.path.join(output_dir, "ml_processed_train.csv"), index=False)
    valid_final.to_csv(os.path.join(output_dir, "ml_processed_valid.csv"), index=False)
    test_final.to_csv(os.path.join(output_dir, "ml_processed_test.csv"), index=False)
    
    # Save feature order for production consistency
    feature_order = train_final.drop(columns=['acuity']).columns.tolist()
    feature_order_path = os.path.join(output_dir, "feature_order.pkl")
    with open(feature_order_path, 'wb') as f:
        pickle.dump(feature_order, f)
    print(f"✓ Saved feature order ({len(feature_order)} features) to: {feature_order_path}")
    
    print(f"\nSaved all datasets to {output_dir}/")

if __name__ == "__main__":
    input_dir = "../raw_data"
    output_dir = "ml_processed_data"
    
    print("Using clipping method to handle impossible values.")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Process training data and fit the scaler
    print("\n--- Processing triage_train.csv ---")
    train_df, fitted_scaler = preprocess_file(os.path.join(input_dir, "triage_train.csv"), None)
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(fitted_scaler, f)
    print(f"✓ Saved scaler to: {scaler_path}")
    
    # Process validation and test data using the SAME scaler
    print("\n--- Processing triage_valid.csv ---")
    valid_df, _ = preprocess_file(os.path.join(input_dir, "triage_valid.csv"), None, scaler=fitted_scaler)
    
    print("\n--- Processing triage_test.csv ---")
    test_df, _ = preprocess_file(os.path.join(input_dir, "triage_test.csv"), None, scaler=fitted_scaler)
    
    # Vectorize text features and save all datasets
    if train_df is not None and valid_df is not None and test_df is not None:
        vectorize_text_and_save(train_df, valid_df, test_df, output_dir)
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
