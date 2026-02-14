import pandas as pd
import os
import glob
import sys

def analyze_cleaned_data(file_path):
    print(f"\nAnalyzing cleaned file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    print("-" * 30)
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    if missing > 0:
        print("Missing values per column:")
        print(df.isnull().sum())
        
    # Check duplicates
    dupes = df.duplicated().sum()
    print(f"Total duplicate rows: {dupes}")
    
    # Range of each column
    print("\nRange of each column (Numeric only):")
    print(f"{'Column':<30} | {'Min':<15} | {'Max':<15}")
    print("-" * 66)
    for col in df.select_dtypes(include=['number']).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col:<30} | {str(min_val):<15} | {str(max_val):<15}")
    
    # Distribution of output column
    # Assuming 'acuity' is the target based on previous context
    target_col = 'acuity'
    if target_col in df.columns:
        print(f"\nDistribution of '{target_col}':")
        print(df[target_col].value_counts().sort_index())
        print("\nPercentages:")
        print((df[target_col].value_counts(normalize=True).sort_index() * 100).round(2))
    else:
        print(f"\nTarget column '{target_col}' not found. Columns are: {df.columns.tolist()}")

if __name__ == "__main__":
    # Create analysis directory
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        
    output_file = os.path.join(analysis_dir, "post_process_report.txt")
    
    # Redirect output to file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        processed_dir = "processed_data"
        csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {processed_dir}")
            # Fallback to check current directory
            if os.path.exists("triage_data_cleaned.csv"):
                analyze_cleaned_data("triage_data_cleaned.csv")
        else:
            print(f"Found {len(csv_files)} processed files to analyze.")
            for file_path in sorted(csv_files):
                analyze_cleaned_data(file_path)
                
    sys.stdout = original_stdout
    print(f"Analysis complete. Report saved to: {output_file}")
