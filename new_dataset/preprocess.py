import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_new_dataset(input_file, output_dir, test_size=0.2, valid_size=0.1, random_state=42):
    """
    Preprocess the patient_priority.csv dataset and split into train, valid, and test sets.
    
    Parameters:
    - input_file: path to patient_priority.csv
    - output_dir: directory to save processed files
    - test_size: proportion of data for test set (default 0.2)
    - valid_size: proportion of data for validation set (default 0.1)
    - random_state: random seed for reproducibility
    """
    
    print("="*70)
    print("PREPROCESSING NEW PATIENT PRIORITY DATASET")
    print("="*70)
    
    # Load the dataset
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify numerical and categorical columns
    print("\n--- Column Types ---")
    # Get numerical columns (excluding the first column which is index)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df.columns[0] in numerical_cols:
        numerical_cols.remove(df.columns[0])  # Remove index column
    
    # Get categorical columns (object/string type)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col}")
    
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"  - {col}: {list(unique_vals[:5])}{'...' if len(unique_vals) > 5 else ''}")
    
    # Display triage distribution before cleaning
    print("\nOriginal triage distribution:")
    print(df['triage'].value_counts(dropna=False))
    
    # 1. Handle missing values in target variable
    print("\n--- Step 1: Handling missing values ---")
    missing_triage = df['triage'].isnull().sum()
    print(f"Missing triage values: {missing_triage}")
    
    # Remove rows with missing triage labels
    df = df.dropna(subset=['triage'])
    print(f"Shape after removing missing triage: {df.shape}")
    
    # 2. Map triage labels to numeric values
    print("\n--- Step 2: Encoding triage labels ---")
    triage_mapping = {
        'red': 1,      # Highest priority (critical, life-threatening)
        'orange': 2,   # High priority (urgent)
        'yellow': 3,   # Medium priority (serious but can wait)
        'green': 4     # Lowest priority (minor, non-urgent)
    }
    df['triage_label'] = df['triage'].map(triage_mapping)
    
    # Check if we have any unmapped values
    unmapped = df[df['triage_label'].isnull()]['triage'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Found unmapped triage values: {unmapped}")
        df = df.dropna(subset=['triage_label'])
        print(f"Shape after removing unmapped triage: {df.shape}")
    
    print(f"Triage label distribution:")
    print(df['triage_label'].value_counts().sort_index())
    
    # 3. Handle missing values in features
    print("\n--- Step 3: Handling missing feature values ---")
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts[missing_counts > 0])
    
    # Fill missing numeric values with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'triage_label' and df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled {col} missing values with median: {median_value}")
    
    # For categorical columns, fill with mode (most frequent value)
    categorical_columns = ['Residence_type', 'smoking_status']
    for col in categorical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_value}")
    
    print(f"Shape after filling missing values: {df.shape}")
    
    # 4. One-Hot Encode categorical variables
    print("\n--- Step 4: One-Hot Encoding categorical variables ---")
    
    # Identify categorical columns for OHE (only text-based categorical)
    categorical_cols = ['Residence_type', 'smoking_status']
    
    print(f"Applying One-Hot Encoding to: {categorical_cols}")
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    
    print(f"Shape after One-Hot Encoding: {df_encoded.shape}")
    print(f"New columns created: {[col for col in df_encoded.columns if col not in df.columns]}")
    
    # 5. Remove duplicate rows
    print("\n--- Step 5: Removing duplicates ---")
    duplicates = df_encoded.duplicated().sum()
    df_encoded = df_encoded.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")
    print(f"Shape after removing duplicates: {df_encoded.shape}")
    
    # 6. Remove first column (index) only
    print("\n--- Step 6: Removing first column ---")
    
    # Drop the first column (unnamed index) and original triage column, keep triage_label
    first_col = df_encoded.columns[0]
    print(f"Removing column: {first_col}")
    
    # Keep all columns except the first one and original 'triage'
    df_final = df_encoded.drop(columns=[first_col, 'triage']).copy()
    
    # Rename triage_label back to triage
    df_final = df_final.rename(columns={'triage_label': 'triage'})
    
    # Move triage column to the last position
    cols = [col for col in df_final.columns if col != 'triage']
    cols.append('triage')
    df_final = df_final[cols]
    
    print(f"Final columns: {df_final.columns.tolist()}")
    print(f"Final shape: {df_final.shape}")
    
    # 7. Split the dataset
    print("\n--- Step 7: Splitting dataset ---")
    
    # First split: separate test set
    train_valid, test = train_test_split(
        df_final, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_final['triage']
    )
    
    # Second split: separate validation from training
    # Adjust valid_size relative to the remaining data
    valid_size_adjusted = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_size_adjusted,
        random_state=random_state,
        stratify=train_valid['triage']
    )
    
    print(f"Train set shape: {train.shape}")
    print(f"Valid set shape: {valid.shape}")
    print(f"Test set shape: {test.shape}")
    
    print("\nTrain set triage distribution:")
    print(train['triage'].value_counts().sort_index())
    print("\nValid set triage distribution:")
    print(valid['triage'].value_counts().sort_index())
    print("\nTest set triage distribution:")
    print(test['triage'].value_counts().sort_index())
    
    # 8. Save the processed datasets
    print("\n--- Step 8: Saving processed datasets ---")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Reset indices to clean sequential numbers
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    valid_path = os.path.join(output_dir, 'valid.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"✓ Saved train set to: {train_path}")
    print(f"✓ Saved valid set to: {valid_path}")
    print(f"✓ Saved test set to: {test_path}")
    
    # Save column names for future predictions
    feature_columns = [col for col in df_final.columns if col != 'triage']
    column_info = {
        'feature_columns': feature_columns,
        'categorical_encoded': {
            'Residence_type': list(df_encoded.filter(like='Residence_type').columns),
            'smoking_status': list(df_encoded.filter(like='smoking_status').columns)
        }
    }
    
    import json
    column_file = os.path.join(output_dir, 'column_info.json')
    with open(column_file, 'w') as f:
        json.dump(column_info, f, indent=2)
    print(f"✓ Saved column info to: {column_file}")
    print(f"\n  Total features: {len(feature_columns)}")
    print(f"  Feature columns: {feature_columns}")
    
    # 9. Display sample data
    print("\n--- Step 9: Sample Data ---")
    print("\nTrain set - First 5 rows:")
    print(train.head())
    print("\nValidation set - First 5 rows:")
    print(valid.head())
    print("\nTest set - First 5 rows:")
    print(test.head())
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    
    # Create a summary
    summary = {
        'original_rows': len(df),
        'processed_rows': len(df_final),
        'train_rows': len(train),
        'valid_rows': len(valid),
        'test_rows': len(test),
        'features': len(df_final.columns) - 1,  # Exclude triage column
        'classes': df_final['triage'].nunique()
    }
    
    return summary


def preprocess_new_data_for_prediction(new_data_file, column_info_file):
    """
    Preprocess new data for prediction using the same encoding as training data.
    
    Parameters:
    - new_data_file: path to new CSV file
    - column_info_file: path to column_info.json saved during training
    
    Returns:
    - DataFrame with same columns as training data
    """
    import json
    
    print("="*70)
    print("PREPROCESSING NEW DATA FOR PREDICTION")
    print("="*70)
    
    # Load column info
    with open(column_info_file, 'r') as f:
        column_info = json.load(f)
    
    expected_features = column_info['feature_columns']
    
    # Load new data
    df = pd.read_csv(new_data_file)
    print(f"New data shape: {df.shape}")
    
    # Remove first column if it's an index
    if df.columns[0] in ['Unnamed: 0', '']:
        df = df.drop(columns=[df.columns[0]])
    
    # Fill missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = ['Residence_type', 'smoking_status']
    for col in categorical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Residence_type', 'smoking_status'], drop_first=False, dtype=int)
    
    # Ensure all expected columns exist (add missing with 0)
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            print(f"Added missing column: {col}")
    
    # Remove extra columns and reorder to match training
    df_final = df_encoded[expected_features]
    
    print(f"Preprocessed data shape: {df_final.shape}")
    print(f"Columns match training: {list(df_final.columns) == expected_features}")
    
    return df_final


if __name__ == "__main__":
    # Configuration
    input_file = "patient_priority.csv"
    output_dir = "processed_data"
    
    # Run preprocessing
    summary = preprocess_new_dataset(
        input_file=input_file,
        output_dir=output_dir,
        test_size=0.2,      # 20% for test
        valid_size=0.1,     # 10% for validation
        random_state=42
    )
    
    print("\n📊 Summary:")
    print(f"  Original rows: {summary['original_rows']}")
    print(f"  Processed rows: {summary['processed_rows']}")
    print(f"  Train rows: {summary['train_rows']} ({summary['train_rows']/summary['processed_rows']*100:.1f}%)")
    print(f"  Valid rows: {summary['valid_rows']} ({summary['valid_rows']/summary['processed_rows']*100:.1f}%)")
    print(f"  Test rows: {summary['test_rows']} ({summary['test_rows']/summary['processed_rows']*100:.1f}%)")
    print(f"  Features: {summary['features']}")
    print(f"  Classes: {summary['classes']}")

