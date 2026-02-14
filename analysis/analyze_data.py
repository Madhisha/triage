import pandas as pd
import glob
import os
import sys

def analyze_file(file_path):
    print("\n" + "#" * 60)
    print(f"ANALYZING FILE: {file_path}")
    print("#" * 60)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # 1. Ignore subject_id and stay_id for analysis
    cols_to_ignore = ['subject_id', 'stay_id']
    existing_ignore_cols = [col for col in cols_to_ignore if col in df.columns]
    
    if existing_ignore_cols:
        print(f"Ignoring columns for duplicate check: {existing_ignore_cols}")
        df_analysis = df.drop(columns=existing_ignore_cols)
    else:
        df_analysis = df
        print("Columns subject_id/stay_id not found, using all columns.")

    # 2. Analyze Duplicates
    # keep=False marks ALL duplicates as True (including the first occurrence)
    # Default (keep='first') would only mark the redundant ones (2529 vs 3054)
    duplicates = df_analysis[df_analysis.duplicated()] 
    
    print("\n" + "-"*50)
    print(f"DUPLICATE ROWS (ignoring {existing_ignore_cols})")
    print("-"*50)
    print(f"Total duplicate rows found: {len(duplicates)}")
    
    if not duplicates.empty:
        print("\nSample of duplicate rows (first 10):")
        # Print the first 10 duplicates with their original index
        print(duplicates.head(10).to_string())
        
        if len(duplicates) > 10:
            print(f"\n... and {len(duplicates) - 10} more duplicate rows.")
    
    # 3. Analyze Acuity 5.0
    print("\n" + "-"*50)
    print("ACUITY 5.0 ROWS")
    print("-"*50)
    
    if 'acuity' in df.columns:
        acuity_5_rows = df[df['acuity'] == 5.0]
        acuity_5_indices = acuity_5_rows.index.tolist()
        
        print(f"Total rows with Acuity 5.0: {len(acuity_5_indices)}")
        if acuity_5_indices:
            print(f"Row numbers (0-based index): {acuity_5_indices}")
    else:
        print("Column 'acuity' not found in dataset.")

    # 4. General Analysis (Missing values, counts) on the subset
    print("\n" + "-"*50)
    print("GENERAL ANALYSIS (on subset)")
    print("-"*50)
    
    # Missing values
    print("\nMissing Values per column:")
    print(df_analysis.isnull().sum())
    
    # Unique values count
    print("\nUnique Values count per column:")
    print(df_analysis.nunique())
    
    # Acuity values specifically
    if 'acuity' in df.columns:
        print("\nAcuity Value Counts:")
        print(df['acuity'].value_counts().sort_index())

def main():
    # Find all triage_*.csv files
    triage_files = glob.glob("raw_data/triage_*.csv")
    
    if not triage_files:
        print("No triage_*.csv files found.")
        return
        
    output_file = "analysis/triage_analysis_report.txt"
    print(f"Found {len(triage_files)} files. Saving analysis to {output_file}...")
    
    original_stdout = sys.stdout
    
    try:
        with open(output_file, 'w') as f:
            sys.stdout = f
            for file_path in sorted(triage_files):
                analyze_file(file_path)
    finally:
        sys.stdout = original_stdout
        
    print(f"Analysis complete. Report saved to: {output_file}")

if __name__ == "__main__":
    main()
