import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define physiologically valid ranges (same as in preprocess.py)
# Values outside these ranges are considered physiologically impossible or extreme outliers
VALID_RANGES = {
    "temperature": (91.4, 107.6),
    "heartrate": (10, 300),
    "resprate": (3, 60),
    "o2sat": (60, 100),
    "sbp": (30, 300),
    "dbp": (30, 300),
    "pain": (0, 10)
}

def detect_physiological_outliers(data, column):
    """
    Detect outliers based on physiological impossibility
    Returns: outlier data, valid min, valid max, outlier count, outlier types
    """
    if column not in VALID_RANGES:
        return pd.DataFrame(), None, None, 0, {}
    
    min_val, max_val = VALID_RANGES[column]
    
    # Outliers: values outside valid physiological range
    outliers = data[(data[column] < min_val) | (data[column] > max_val)]
    
    # Count by type
    below_min = len(data[data[column] < min_val])
    above_max = len(data[data[column] > max_val])
    
    outlier_types = {
        'below_minimum': below_min,
        'above_maximum': above_max,
        'total': len(outliers)
    }
    
    return outliers, min_val, max_val, len(outliers), outlier_types


def detect_outliers_iqr(data, column):
    """
    Detect outliers using IQR method (for comparison)
    Returns: outlier indices, lower bound, upper bound, outlier count
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound, len(outliers)

def analyze_dataset_outliers(df, dataset_name):
    """
    Analyze outliers in a dataset and print statistics
    """
    print(f"\n{'='*80}")
    print(f"OUTLIER ANALYSIS FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Total records: {len(df)}")
    
    # Numeric columns to analyze
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    # Ensure all columns are numeric
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    outlier_summary = {}
    
    for col in numeric_columns:
        if col in df.columns:
            print(f"\n{'-'*80}")
            print(f"Column: {col.upper()}")
            print(f"{'-'*80}")
            
            # Basic statistics
            print(f"Mean: {df[col].mean():.2f}")
            print(f"Median: {df[col].median():.2f}")
            print(f"Std Dev: {df[col].std():.2f}")
            print(f"Min: {df[col].min():.2f}")
            print(f"Max: {df[col].max():.2f}")
            
            # Physiological outliers (MAIN METHOD)
            outliers_phys, valid_min, valid_max, count_phys, types = detect_physiological_outliers(df, col)
            print(f"\n*** PHYSIOLOGICALLY IMPOSSIBLE VALUES ***")
            print(f"  Valid range: [{valid_min}, {valid_max}]")
            print(f"  Below minimum ({valid_min}): {types['below_minimum']} ({types['below_minimum']/len(df)*100:.2f}%)")
            print(f"  Above maximum ({valid_max}): {types['above_maximum']} ({types['above_maximum']/len(df)*100:.2f}%)")
            print(f"  TOTAL OUTLIERS: {count_phys} ({count_phys/len(df)*100:.2f}%)")
            
            outlier_summary[col] = {
                'phys_count': count_phys,
                'phys_percentage': count_phys/len(df)*100,
                'phys_below': types['below_minimum'],
                'phys_above': types['above_maximum'],
                'valid_min': valid_min,
                'valid_max': valid_max
            }
    
    return outlier_summary

def plot_outliers_single_dataset(df, dataset_name, save_path):
    """
    Create outlier visualizations for a single dataset using physiological ranges
    """
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    available_columns = [col for col in numeric_columns if col in df.columns]
    
    n_cols = len(available_columns)
    fig, axes = plt.subplots(2, n_cols, figsize=(22, 10))
    fig.suptitle(f'Outlier Analysis - {dataset_name} (Physiological Ranges)', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(available_columns):
        # Box plot
        ax1 = axes[0, idx] if n_cols > 1 else axes[0]
        ax1.boxplot(df[col].dropna(), vert=True)
        
        # Add physiological range lines
        if col in VALID_RANGES:
            valid_min, valid_max = VALID_RANGES[col]
            ax1.axhline(valid_min, color='red', linestyle='--', linewidth=2, label='Valid Range', alpha=0.7)
            ax1.axhline(valid_max, color='red', linestyle='--', linewidth=2, alpha=0.7)

        
        # Histogram with range boundaries
        ax2 = axes[1, idx] if n_cols > 1 else axes[1]
        ax2.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        
        # Add physiological boundaries
        if col in VALID_RANGES:
            valid_min, valid_max = VALID_RANGES[col]
            ax2.axvline(valid_min, color='red', linestyle='--', linewidth=2, label='Valid Range')
            ax2.axvline(valid_max, color='red', linestyle='--', linewidth=2)
                
            # Count outliers
            outliers, _, _, count, _ = detect_physiological_outliers(df, col)
            ax2.text(0.98, 0.95, f'Outliers: {count}\n({count/len(df)*100:.1f}%)', 
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
        
        ax2.set_title(f'{col} Distribution', fontweight='bold')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.show()

def plot_outlier_comparison(train_summary, test_summary, valid_summary):
    """
    Create comparison bar chart for physiological outlier percentages across datasets
    """
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Physiologically Impossible Values - Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    # Physiological outliers comparison
    x = np.arange(len(numeric_columns))
    width = 0.25
    
    train_phys = [train_summary[col]['phys_percentage'] for col in numeric_columns]
    test_phys = [test_summary[col]['phys_percentage'] for col in numeric_columns]
    valid_phys = [valid_summary[col]['phys_percentage'] for col in numeric_columns]
    
    ax.bar(x - width, train_phys, width, label='Train', color='blue', alpha=0.7)
    ax.bar(x, test_phys, width, label='Test', color='green', alpha=0.7)
    ax.bar(x + width, valid_phys, width, label='Validation', color='orange', alpha=0.7)
    
    ax.set_xlabel('Variables', fontweight='bold', fontsize=12)
    ax.set_ylabel('Outlier Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Percentage of Physiologically Impossible Values by Variable', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(numeric_columns, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outlier_comparison_chart.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved: outlier_comparison_chart.png")
    plt.show()

def plot_combined_outliers(train_df, test_df, valid_df):
    """
    Create combined outlier visualization for all datasets with physiological ranges
    """
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    fig, axes = plt.subplots(3, len(numeric_columns), figsize=(26, 13))
    fig.suptitle('Combined Outlier Analysis - All Datasets (Physiological Ranges)', fontsize=18, fontweight='bold')
    
    datasets = [
        (train_df, 'Train', 'blue'),
        (test_df, 'Test', 'green'),
        (valid_df, 'Validation', 'orange')
    ]
    
    for row_idx, (df, name, color) in enumerate(datasets):
        for col_idx, col in enumerate(numeric_columns):
            ax = axes[row_idx, col_idx]
            
            # Box plot
            bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.5),
                           medianprops=dict(color='red', linewidth=2))
            
            # Add physiological range lines
            if col in VALID_RANGES:
                valid_min, valid_max = VALID_RANGES[col]
                ax.axhline(valid_min, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
                ax.axhline(valid_max, color='darkred', linestyle='--', linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{col}\n({name})', fontweight='bold', fontsize=10)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add outlier count
            outliers, _, _, count, types = detect_physiological_outliers(df, col)
            info_text = f'Impossible: {count}\nBelow: {types["below_minimum"]}\nAbove: {types["above_maximum"]}'
            ax.text(0.5, 0.02, info_text, 
                   transform=ax.transAxes, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=7)
    
    plt.tight_layout()
    plt.savefig('outlier_analysis_combined.png', dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved: outlier_analysis_combined.png")
    plt.show()

def plot_outlier_comparison(train_summary, test_summary, valid_summary):
    """
    Create comparison bar chart for physiological outlier percentages across datasets
    """
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Physiologically Impossible Values - Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    # Physiological outliers comparison
    x = np.arange(len(numeric_columns))
    width = 0.25
    
    train_phys = [train_summary[col]['phys_percentage'] for col in numeric_columns]
    test_phys = [test_summary[col]['phys_percentage'] for col in numeric_columns]
    valid_phys = [valid_summary[col]['phys_percentage'] for col in numeric_columns]
    
    ax.bar(x - width, train_phys, width, label='Train', color='blue', alpha=0.7)
    ax.bar(x, test_phys, width, label='Test', color='green', alpha=0.7)
    ax.bar(x + width, valid_phys, width, label='Validation', color='orange', alpha=0.7)
    
    ax.set_xlabel('Variables', fontweight='bold', fontsize=12)
    ax.set_ylabel('Outlier Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Percentage of Physiologically Impossible Values by Variable', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(numeric_columns, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outlier_comparison_chart.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved: outlier_comparison_chart.png")
    plt.show()

def main():
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS - RAW TRIAGE DATASET")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv('../raw_data/triage_train.csv')
    test_df = pd.read_csv('../raw_data/triage_test.csv')
    valid_df = pd.read_csv('../raw_data/triage_valid.csv')
    print(f"Train: {len(train_df)} records")
    print(f"Test: {len(test_df)} records")
    print(f"Validation: {len(valid_df)} records")
    
    # Analyze each dataset
    train_summary = analyze_dataset_outliers(train_df, "TRAIN")
    test_summary = analyze_dataset_outliers(test_df, "TEST")
    valid_summary = analyze_dataset_outliers(valid_df, "VALIDATION")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL OUTLIER SUMMARY - PHYSIOLOGICALLY IMPOSSIBLE VALUES")
    print(f"{'='*80}")
    
    numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    for col in numeric_columns:
        print(f"\n{col.upper()}:")
        print(f"  Valid Range: [{VALID_RANGES[col][0]}, {VALID_RANGES[col][1]}]")
        print(f"  Train    - Impossible: {train_summary[col]['phys_count']:4d} ({train_summary[col]['phys_percentage']:5.2f}%)")
        print(f"  Test     - Impossible: {test_summary[col]['phys_count']:4d} ({test_summary[col]['phys_percentage']:5.2f}%)")
        print(f"  Valid    - Impossible: {valid_summary[col]['phys_count']:4d} ({valid_summary[col]['phys_percentage']:5.2f}%)")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Only train dataset plot
    plot_outliers_single_dataset(train_df, "TRAIN Dataset", "outlier_analysis_train.png")
    
    # Comparison chart
    plot_outlier_comparison(train_summary, test_summary, valid_summary)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  1. outlier_analysis_train.png")
    print("  2. outlier_comparison_chart.png")
    print("\n")

if __name__ == "__main__":
    main()
