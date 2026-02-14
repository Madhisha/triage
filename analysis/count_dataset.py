import pandas as pd
import os
import sys

class Tee:
    """Write to both file and console"""
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

def count_dataset_stats():
    """Count and display statistics for raw train, validation, and test datasets"""
    
    # Redirect output to both console and file
    tee = Tee("count.txt")
    sys.stdout = tee
    
    # Define file paths
    train_file = "raw_data/triage_train.csv"
    valid_file = "raw_data/triage_valid.csv"
    test_file = "raw_data/triage_test.csv"
    
    # Check if files exist
    for file in [train_file, valid_file, test_file]:
        if not os.path.exists(file):
            print(f"Error: {file} not found!")
            sys.stdout = tee.stdout
            tee.close()
            return
    
    print("="*70)
    print("DATASET STATISTICS - RAW DATA")
    print("="*70)
    
    # Load datasets
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    
    # Training Set
    print("\n" + "-"*70)
    print("TRAINING SET (triage_train.csv)")
    print("-"*70)
    print(f"Total Records: {len(train_df):,}")
    print(f"Total Columns: {len(train_df.columns)}")
    print(f"\nClass Distribution (Acuity):")
    if 'acuity' in train_df.columns:
        acuity_counts = train_df['acuity'].value_counts().sort_index()
        for acuity, count in acuity_counts.items():
            print(f"  Class {acuity}: {count:,}")
        print(f"\nMissing Acuity Values: {train_df['acuity'].isna().sum():,}")
    
    # Validation Set
    print("\n" + "-"*70)
    print("VALIDATION SET (triage_valid.csv)")
    print("-"*70)
    print(f"Total Records: {len(valid_df):,}")
    print(f"Total Columns: {len(valid_df.columns)}")
    print(f"\nClass Distribution (Acuity):")
    if 'acuity' in valid_df.columns:
        acuity_counts = valid_df['acuity'].value_counts().sort_index()
        for acuity, count in acuity_counts.items():
            print(f"  Class {acuity}: {count:,}")
        print(f"\nMissing Acuity Values: {valid_df['acuity'].isna().sum():,}")
    
    # Test Set
    print("\n" + "-"*70)
    print("TEST SET (triage_test.csv)")
    print("-"*70)
    print(f"Total Records: {len(test_df):,}")
    print(f"Total Columns: {len(test_df.columns)}")
    print(f"\nClass Distribution (Acuity):")
    if 'acuity' in test_df.columns:
        acuity_counts = test_df['acuity'].value_counts().sort_index()
        for acuity, count in acuity_counts.items():
            print(f"  Class {acuity}: {count:,}")
        print(f"\nMissing Acuity Values: {test_df['acuity'].isna().sum():,}")
    
    # Overall Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    total_records = len(train_df) + len(valid_df) + len(test_df)
    print(f"Total Records Across All Sets: {total_records:,}")
    print(f"  - Training:   {len(train_df):,} ({len(train_df)/total_records*100:.1f}%)")
    print(f"  - Validation: {len(valid_df):,} ({len(valid_df)/total_records*100:.1f}%)")
    print(f"  - Test:       {len(test_df):,} ({len(test_df)/total_records*100:.1f}%)")
    
    # Check for duplicates
    print("\n" + "-"*70)
    print("DUPLICATE ANALYSIS")
    print("-"*70)
    cols_to_check = [col for col in train_df.columns if col not in ['subject_id', 'stay_id']]
    print(f"Checking duplicates (excluding subject_id, stay_id):")
    print(f"  - Training duplicates:   {train_df[cols_to_check].duplicated().sum():,}")
    print(f"  - Validation duplicates: {valid_df[cols_to_check].duplicated().sum():,}")
    print(f"  - Test duplicates:       {test_df[cols_to_check].duplicated().sum():,}")
    
    # Missing values summary
    print("\n" + "-"*70)
    print("MISSING VALUES SUMMARY")
    print("-"*70)
    print("\nTraining Set:")
    print(train_df.isnull().sum())
    print("\nValidation Set:")
    print(valid_df.isnull().sum())
    print("\nTest Set:")
    print(test_df.isnull().sum())
    
    print("\n" + "="*70)
    
    # Restore stdout and close file
    sys.stdout = tee.stdout
    tee.close()
    print("\n✓ Results saved to count.txt")

if __name__ == "__main__":
    count_dataset_stats()
