"""
Comprehensive Exploratory Data Analysis (EDA) for Patient Triage Dataset
Generates visualizations and text report covering all aspects of the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
import os
import sys
from collections import Counter
import re

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get the directory where this script is located (analysis folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create output directories in the same folder as this script
PLOTS_DIR = os.path.join(SCRIPT_DIR, "eda_plots")
REPORT_FILE = os.path.join(SCRIPT_DIR, "eda_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)


class Tee:
    """Write to both file and console"""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()


def print_header(title, char="="):
    """Print a formatted header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


def print_subheader(title, char="-"):
    """Print a formatted subheader"""
    print(f"\n{char * 70}")
    print(f"{title}")
    print(f"{char * 70}\n")


def load_datasets():
    """Load train, validation, and test datasets"""
    print_header("LOADING DATASETS")
    
    train_df = pd.read_csv("../raw_data/triage_train.csv")
    valid_df = pd.read_csv("../raw_data/triage_valid.csv")
    test_df = pd.read_csv("../raw_data/triage_test.csv")
    
    print(f"Training set:   {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"Validation set: {valid_df.shape[0]:,} rows × {valid_df.shape[1]} columns")
    print(f"Test set:       {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
    print(f"Total:          {train_df.shape[0] + valid_df.shape[0] + test_df.shape[0]:,} rows")
    
    return train_df, valid_df, test_df


def analyze_basic_info(df, dataset_name):
    """Analyze basic dataset information"""
    print_subheader(f"1. BASIC INFORMATION - {dataset_name}")
    
    print("Dataset Shape:", df.shape)
    print("\nColumn Names and Types:")
    print(df.dtypes)
    print("\nMemory Usage:")
    print(df.memory_usage(deep=True))
    print(f"\nTotal Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def analyze_missing_values(df, dataset_name):
    """Analyze missing values with visualization"""
    print_subheader(f"2. MISSING VALUES ANALYSIS - {dataset_name}")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    missing_df_filtered = missing_df[missing_df['Missing_Count'] > 0]
    if not missing_df_filtered.empty:
        bars1 = axes[0].barh(missing_df_filtered['Column'], missing_df_filtered['Missing_Count'], color='salmon')
        axes[0].set_xlabel('Missing Count')
        axes[0].set_title(f'Missing Values Count - {dataset_name}')
        axes[0].invert_yaxis()
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, missing_df_filtered['Missing_Count'])):
            axes[0].text(val, bar.get_y() + bar.get_height()/2, f' {int(val)}', 
                        va='center', fontsize=9, fontweight='bold')
        
        # Percentage plot
        bars2 = axes[1].barh(missing_df_filtered['Column'], missing_df_filtered['Missing_Percentage'], 
                    color='lightcoral')
        axes[1].set_xlabel('Missing Percentage (%)')
        axes[1].set_title(f'Missing Values Percentage - {dataset_name}')
        axes[1].invert_yaxis()
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, missing_df_filtered['Missing_Percentage'])):
            axes[1].text(val, bar.get_y() + bar.get_height()/2, f' {val:.2f}%', 
                        va='center', fontsize=9, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No missing values', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No missing values', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/missing_values_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: missing_values_{dataset_name.lower()}.png")


def analyze_duplicates(df, dataset_name):
    """Analyze duplicate rows"""
    print_subheader(f"3. DUPLICATE ANALYSIS - {dataset_name}")
    
    # Check duplicates ignoring ID columns
    cols_to_ignore = ['subject_id', 'stay_id']
    df_check = df.drop(columns=[c for c in cols_to_ignore if c in df.columns], errors='ignore')
    
    duplicates = df_check.duplicated()
    dup_count = duplicates.sum()
    
    print(f"Total duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    
    if dup_count > 0:
        print(f"\nFirst 10 duplicate rows:")
        print(df[duplicates].head(10).to_string())


def analyze_target_distribution(train_df, valid_df, test_df):
    """Analyze target variable (acuity) distribution"""
    print_subheader("4. TARGET VARIABLE DISTRIBUTION (ACUITY)")
    
    # Combined analysis
    print("Class Distribution Across Splits:\n")
    
    all_data = []
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        counts = df['acuity'].value_counts().sort_index()
        for acuity, count in counts.items():
            all_data.append({
                'Split': name,
                'Acuity': int(acuity) if pd.notna(acuity) else 'Missing',
                'Count': count,
                'Percentage': count / len(df) * 100
            })
    
    dist_df = pd.DataFrame(all_data)
    
    # Print by acuity level
    for acuity in sorted(train_df['acuity'].dropna().unique()):
        print(f"\nAcuity Level {int(acuity)}:")
        subset = dist_df[dist_df['Acuity'] == int(acuity)]
        for _, row in subset.iterrows():
            print(f"  {row['Split']:8s}: {row['Count']:6,} ({row['Percentage']:5.2f}%)")
    
    # Check for missing acuity
    print(f"\nMissing Acuity Values:")
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        missing = df['acuity'].isnull().sum()
        print(f"  {name:8s}: {missing:6,} ({missing/len(df)*100:.2f}%)")
    
    # Visualization - Multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Stacked bar chart by split
    pivot_df = dist_df[dist_df['Acuity'] != 'Missing'].pivot(
        index='Split', columns='Acuity', values='Count'
    ).fillna(0)
    
    pivot_df.plot(kind='bar', stacked=False, ax=axes[0, 0], 
                 color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6'])
    axes[0, 0].set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Split')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend(title='Acuity', bbox_to_anchor=(1.05, 1))
    axes[0, 0].tick_params(axis='x', rotation=0)
    # Add value labels on bars
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt='%d', fontsize=8, fontweight='bold')
    
    # Plot 2: Pie chart for training data
    train_counts = train_df['acuity'].value_counts().sort_index()
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    axes[0, 1].pie(train_counts.values, labels=[f'Acuity {int(x)}' for x in train_counts.index],
                   autopct='%1.1f%%', colors=colors[:len(train_counts)], startangle=90)
    axes[0, 1].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Percentage stacked bar
    pct_pivot = dist_df[dist_df['Acuity'] != 'Missing'].pivot(
        index='Split', columns='Acuity', values='Percentage'
    ).fillna(0)
    
    pct_pivot.plot(kind='bar', stacked=True, ax=axes[1, 0],
                   color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6'])
    axes[1, 0].set_title('Class Distribution (Percentage) by Split', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Split')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend(title='Acuity', bbox_to_anchor=(1.05, 1))
    axes[1, 0].tick_params(axis='x', rotation=0)
    # Add value labels on stacked bars
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt='%.1f%%', label_type='center', fontsize=7, fontweight='bold')
    
    # Plot 4: Combined histogram
    combined_df = pd.concat([
        train_df[['acuity']].assign(split='Train'),
        valid_df[['acuity']].assign(split='Valid'),
        test_df[['acuity']].assign(split='Test')
    ])
    
    for split, color in [('Train', '#3498db'), ('Valid', '#f39c12'), ('Test', '#e74c3c')]:
        data = combined_df[combined_df['split'] == split]['acuity'].dropna()
        axes[1, 1].hist(data, bins=5, alpha=0.6, label=split, color=color, edgecolor='black')
    
    axes[1, 1].set_title('Acuity Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Acuity Level')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_xticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/target_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: target_distribution.png")
    
    # Imbalance ratio
    print("\nClass Imbalance Ratio (Training Set):")
    max_class = train_counts.max()
    for acuity, count in train_counts.items():
        ratio = max_class / count
        print(f"  Acuity {int(acuity)}: 1:{ratio:.2f}")


def analyze_numerical_features(df, dataset_name):
    """Comprehensive numerical features analysis"""
    print_subheader(f"5. NUMERICAL FEATURES ANALYSIS - {dataset_name}")
    
    numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    # Descriptive statistics
    print("Descriptive Statistics:\n")
    stats_df = df[numerical_cols].describe()
    print(stats_df.to_string())
    
    # Additional statistics
    print("\n\nAdditional Statistics:")
    print(f"{'Feature':<15} {'Skewness':>10} {'Kurtosis':>10} {'Missing':>10}")
    print("-" * 50)
    for col in numerical_cols:
        # Convert to numeric, coercing errors (like 'c' in pain) to NaN
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(data) > 0:
            skew = stats.skew(data)
            kurt = stats.kurtosis(data)
            missing = df[col].isnull().sum()
            print(f"{col:<15} {skew:>10.3f} {kurt:>10.3f} {missing:>10,}")
    
    # Visualizations
    # 1. Distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        # Convert to numeric, handling any non-numeric values
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(data) > 0:
            # Histogram with KDE
            axes[idx].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            
            # Add KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
            
            axes[idx].set_title(f'{col.capitalize()} Distribution', fontweight='bold')
            axes[idx].set_xlabel(col.capitalize())
            axes[idx].set_ylabel('Density')
            axes[idx].grid(True, alpha=0.3)
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            axes[idx].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            axes[idx].text(0.95, 0.95, f'Std: {std_val:.2f}', transform=axes[idx].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
            axes[idx].legend(fontsize=8)
    
    # Remove extra subplot
    if len(numerical_cols) < len(axes):
        fig.delaxes(axes[-1])
        fig.delaxes(axes[-2])
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/numerical_distributions_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: numerical_distributions_{dataset_name.lower()}.png")
    
    # 2. Box plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        # Convert to numeric, handling any non-numeric values
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(data) > 0:
            bp = axes[idx].boxplot(data, vert=True, patch_artist=True, 
                                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                                   medianprops=dict(color='red', linewidth=2),
                                   whiskerprops=dict(linewidth=1.5),
                                   capprops=dict(linewidth=1.5))
            
            axes[idx].set_title(f'{col.capitalize()} Box Plot', fontweight='bold')
            axes[idx].set_ylabel(col.capitalize())
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add statistics text in a clean text box (removed overlapping labels)
            q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
            min_val, max_val = data.min(), data.max()
            iqr = q3 - q1
            stats_text = f'Min: {min_val:.1f}\nQ1: {q1:.1f}\nMedian: {median:.1f}\nQ3: {q3:.1f}\nMax: {max_val:.1f}\nIQR: {iqr:.1f}'
            axes[idx].text(0.98, 0.98, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                          fontsize=9)
    
    if len(numerical_cols) < len(axes):
        fig.delaxes(axes[-1])
        fig.delaxes(axes[-2])
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/numerical_boxplots_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot: numerical_boxplots_{dataset_name.lower()}.png")


def analyze_outliers(df, dataset_name):
    """Detect and analyze outliers"""
    print_subheader(f"6. OUTLIER ANALYSIS - {dataset_name}")
    
    # Define physiologically valid ranges
    valid_ranges = {
        "temperature": (91.4, 107.6),
        "heartrate": (10, 300),
        "resprate": (3, 60),
        "o2sat": (60, 100),
        "sbp": (30, 300),
        "dbp": (30, 300),
        "pain": (0, 10)
    }
    
    print("Outlier Detection (Physiological Range + IQR Method):\n")
    print(f"{'Feature':<15} {'Physio_Out':>12} {'IQR_Out':>12} {'Total_Valid':>12} {'Outlier_%':>12}")
    print("-" * 70)
    
    outlier_data = []
    
    for col in valid_ranges.keys():
        if col in df.columns:
            # Convert to numeric, handling any non-numeric values
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            
            # Physiological outliers
            min_val, max_val = valid_ranges[col]
            physio_out = ((data < min_val) | (data > max_val)).sum()
            
            # IQR outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_out = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            
            total_valid = len(data)
            outlier_pct = (physio_out / total_valid * 100) if total_valid > 0 else 0
            
            print(f"{col:<15} {physio_out:>12,} {iqr_out:>12,} {total_valid:>12,} {outlier_pct:>11.2f}%")
            
            outlier_data.append({
                'Feature': col,
                'Physiological': physio_out,
                'IQR': iqr_out
            })
    
    # Visualization
    outlier_df = pd.DataFrame(outlier_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(outlier_df))
    width = 0.35
    
    bars1 = axes[0].bar(x_pos - width/2, outlier_df['Physiological'], width, 
               label='Physiological', color='salmon', alpha=0.8)
    bars2 = axes[0].bar(x_pos + width/2, outlier_df['IQR'], width, 
               label='IQR Method', color='lightblue', alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(outlier_df['Feature'], rotation=45, ha='right')
    axes[0].set_ylabel('Number of Outliers')
    axes[0].set_title(f'Outlier Detection Comparison - {dataset_name}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Percentage view
    total_counts = [len(pd.to_numeric(df[col], errors='coerce').dropna()) for col in outlier_df['Feature']]
    physio_pct = (outlier_df['Physiological'] / total_counts) * 100
    iqr_pct = (outlier_df['IQR'] / total_counts) * 100
    
    bars3 = axes[1].bar(x_pos - width/2, physio_pct, width, 
               label='Physiological', color='salmon', alpha=0.8)
    bars4 = axes[1].bar(x_pos + width/2, iqr_pct, width, 
               label='IQR Method', color='lightblue', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(outlier_df['Feature'], rotation=45, ha='right')
    axes[1].set_ylabel('Percentage of Outliers (%)')
    axes[1].set_title(f'Outlier Percentage - {dataset_name}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/outliers_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: outliers_{dataset_name.lower()}.png")


def analyze_correlation(df, dataset_name):
    """Analyze correlations between numerical features"""
    print_subheader(f"7. CORRELATION ANALYSIS - {dataset_name}")
    
    numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Convert all columns to numeric, coercing errors to NaN
    df_numeric = df[numerical_cols].copy()
    for col in numerical_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    corr_df = df_numeric.corr()
    
    print("Correlation Matrix:\n")
    print(corr_df.to_string())
    
    # Find strong correlations
    print("\n\nStrong Correlations (|r| > 0.3):")
    strong_corr = []
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) > 0.3:
                strong_corr.append({
                    'Feature 1': corr_df.columns[i],
                    'Feature 2': corr_df.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corr:
        strong_df = pd.DataFrame(strong_corr).sort_values('Correlation', 
                                                          key=abs, ascending=False)
        print(strong_df.to_string(index=False))
    else:
        print("No strong correlations found.")
    
    # Visualization - Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title(f'Correlation Matrix - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_matrix_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: correlation_matrix_{dataset_name.lower()}.png")


def analyze_features_by_target(df, dataset_name):
    """Analyze how features vary across target classes"""
    print_subheader(f"8. FEATURE ANALYSIS BY TARGET CLASS - {dataset_name}")
    
    numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    # Convert all numerical columns to numeric, handling non-numeric values
    df_numeric = df.copy()
    for col in numerical_cols:
        df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Mean Values by Acuity Level:\n")
    
    grouped = df_numeric.groupby('acuity')[numerical_cols].mean()
    print(grouped.to_string())
    
    print("\n\nMedian Values by Acuity Level:\n")
    grouped_median = df_numeric.groupby('acuity')[numerical_cols].median()
    print(grouped_median.to_string())
    
    # Statistical significance tests (ANOVA)
    print("\n\nANOVA Test Results (p-values):")
    print("Testing if feature means differ significantly across acuity levels\n")
    print(f"{'Feature':<15} {'F-statistic':>15} {'p-value':>15} {'Significant':>15}")
    print("-" * 65)
    
    for col in numerical_cols:
        # Convert to numeric for each group to handle non-numeric values
        groups = [pd.to_numeric(df[df['acuity'] == acuity][col], errors='coerce').dropna() 
                 for acuity in df['acuity'].dropna().unique()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            significant = "Yes" if p_value < 0.05 else "No"
            print(f"{col:<15} {f_stat:>15.4f} {p_value:>15.6f} {significant:>15}")
    
    # Visualizations - Box plots by acuity
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        data_to_plot = []
        labels = []
        
        for acuity in sorted(df['acuity'].dropna().unique()):
            # Convert to numeric to handle non-numeric values
            data = pd.to_numeric(df[df['acuity'] == acuity][col], errors='coerce').dropna()
            if len(data) > 0:
                data_to_plot.append(data)
                labels.append(f'Acuity {int(acuity)}')
        
        if data_to_plot:
            bp = axes[idx].boxplot(data_to_plot, labels=labels, patch_artist=True,
                                  boxprops=dict(alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2))
            
            # Color by acuity
            colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            
            # Add median values as text labels (positioned above to avoid overlap)
            medians = [np.median(data) for data in data_to_plot]
            max_vals = [data.max() for data in data_to_plot]
            for i, (pos, median, max_val) in enumerate(zip(range(1, len(medians) + 1), medians, max_vals)):
                y_pos = max_val + (max_val - median) * 0.05  # Position slightly above max
                axes[idx].text(pos, y_pos, f'{median:.1f}', 
                             ha='center', va='bottom', fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='black'))
            
            axes[idx].set_title(f'{col.capitalize()} by Acuity', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Acuity Level')
            axes[idx].set_ylabel(col.capitalize())
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)
    
    # Remove extra subplots
    if len(numerical_cols) < len(axes):
        for idx in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/features_by_acuity_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: features_by_acuity_{dataset_name.lower()}.png")


def analyze_text_features(df, dataset_name):
    """Analyze chief complaint text feature"""
    print_subheader(f"9. TEXT FEATURE ANALYSIS (CHIEF COMPLAINT) - {dataset_name}")
    
    if 'chiefcomplaint' not in df.columns:
        print("Chief complaint column not found.")
        return
    
    complaints = df['chiefcomplaint'].dropna()
    
    print(f"Total Complaints: {len(complaints):,}")
    print(f"Unique Complaints: {complaints.nunique():,}")
    print(f"Missing Complaints: {df['chiefcomplaint'].isnull().sum():,}")
    
    # Length analysis
    complaint_lengths = complaints.str.len()
    word_counts = complaints.str.split().str.len()
    
    print(f"\nComplaint Length Statistics:")
    print(f"  Mean characters: {complaint_lengths.mean():.1f}")
    print(f"  Median characters: {complaint_lengths.median():.1f}")
    print(f"  Max characters: {complaint_lengths.max()}")
    print(f"  Min characters: {complaint_lengths.min()}")
    
    print(f"\nWord Count Statistics:")
    print(f"  Mean words: {word_counts.mean():.1f}")
    print(f"  Median words: {word_counts.median():.1f}")
    print(f"  Max words: {word_counts.max()}")
    print(f"  Min words: {word_counts.min()}")
    
    # Most common complaints
    print(f"\nTop 20 Most Common Chief Complaints:")
    top_complaints = complaints.value_counts().head(20)
    for idx, (complaint, count) in enumerate(top_complaints.items(), 1):
        pct = count / len(complaints) * 100
        print(f"  {idx:2d}. {complaint[:50]:<50s} | {count:5,} ({pct:4.2f}%)")
    
    # Word frequency analysis
    print(f"\nTop 30 Most Common Words in Chief Complaints:")
    all_words = ' '.join(complaints.str.lower()).split()
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
    word_freq = Counter(filtered_words)
    
    for idx, (word, count) in enumerate(word_freq.most_common(30), 1):
        pct = count / len(filtered_words) * 100
        print(f"  {idx:2d}. {word:<20s} | {count:6,} ({pct:4.2f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Length distribution
    axes[0, 0].hist(complaint_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(complaint_lengths.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {complaint_lengths.mean():.1f}')
    axes[0, 0].set_title('Chief Complaint Character Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Characters')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Word count distribution
    axes[0, 1].hist(word_counts, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(word_counts.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {word_counts.mean():.1f}')
    axes[0, 1].set_title('Chief Complaint Word Count Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Top complaints bar chart
    top_15 = complaints.value_counts().head(15)
    bars = axes[1, 0].barh(range(len(top_15)), top_15.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_yticks(range(len(top_15)))
    axes[1, 0].set_yticklabels([c[:30] + '...' if len(c) > 30 else c for c in top_15.index], 
                               fontsize=9)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Top 15 Most Common Chief Complaints', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_15.values)):
        axes[1, 0].text(val, bar.get_y() + bar.get_height()/2, f' {int(val)}', 
                       va='center', fontsize=8, fontweight='bold')
    
    # 4. Top words bar chart
    top_words_20 = word_freq.most_common(20)
    words, counts = zip(*top_words_20)
    bars_words = axes[1, 1].barh(range(len(words)), counts, color='plum', edgecolor='black')
    axes[1, 1].set_yticks(range(len(words)))
    axes[1, 1].set_yticklabels(words, fontsize=9)
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_title('Top 20 Most Common Words', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_words, counts)):
        axes[1, 1].text(val, bar.get_y() + bar.get_height()/2, f' {int(val)}', 
                       va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/text_analysis_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plot: text_analysis_{dataset_name.lower()}.png")


def analyze_text_by_acuity(df, dataset_name):
    """Analyze chief complaints by acuity level"""
    print_subheader(f"10. CHIEF COMPLAINT BY ACUITY - {dataset_name}")
    
    if 'chiefcomplaint' not in df.columns or 'acuity' not in df.columns:
        print("Required columns not found.")
        return
    
    print("Average Complaint Length by Acuity:\n")
    
    for acuity in sorted(df['acuity'].dropna().unique()):
        complaints = df[df['acuity'] == acuity]['chiefcomplaint'].dropna()
        if len(complaints) > 0:
            avg_len = complaints.str.len().mean()
            avg_words = complaints.str.split().str.len().mean()
            print(f"Acuity {int(acuity)}: {avg_len:.1f} chars, {avg_words:.1f} words (n={len(complaints):,})")
    
    # Most common complaints per acuity
    print("\n\nTop 10 Chief Complaints by Acuity Level:\n")
    
    for acuity in sorted(df['acuity'].dropna().unique()):
        print(f"\n--- Acuity {int(acuity)} ---")
        complaints = df[df['acuity'] == acuity]['chiefcomplaint'].dropna()
        top_10 = complaints.value_counts().head(10)
        
        for idx, (complaint, count) in enumerate(top_10.items(), 1):
            pct = count / len(complaints) * 100
            print(f"  {idx:2d}. {complaint[:45]:<45s} | {count:4,} ({pct:5.2f}%)")


def analyze_pairwise_relationships(df, dataset_name):
    """Analyze pairwise relationships between key features"""
    print_subheader(f"11. PAIRWISE FEATURE RELATIONSHIPS - {dataset_name}")
    
    # Select key features for scatter plot (including DBP)
    key_features = ['heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'temperature']
    key_features = [f for f in key_features if f in df.columns]
    
    if len(key_features) < 2:
        print("Not enough features for pairwise analysis.")
        return
    
    print(f"Analyzing pairwise relationships for: {', '.join(key_features)}")
    print("Generating scatter plot matrix...")
    
    # Create scatter plot matrix - convert all to numeric first
    df_subset = df[key_features + ['acuity']].copy()
    for col in key_features + ['acuity']:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
    df_subset = df_subset.dropna()
    
    if len(df_subset) > 0:
        # Sample if too large, but ensure all acuity levels are represented
        if len(df_subset) > 5000:
            # Sample proportionally from each acuity level to maintain all classes
            sampled_dfs = []
            for acuity_val in sorted(df_subset['acuity'].unique()):
                acuity_df = df_subset[df_subset['acuity'] == acuity_val]
                n_samples = min(len(acuity_df), max(50, int(5000 * len(acuity_df) / len(df_subset))))
                sampled_dfs.append(acuity_df.sample(n_samples, random_state=42))
            df_subset = pd.concat(sampled_dfs, ignore_index=True)
            print(f"(Sampled {len(df_subset)} points with all acuity levels: {sorted(df_subset['acuity'].unique())})")
        
        # Create pair plot
        g = sns.pairplot(df_subset, hue='acuity', palette='Set1', 
                        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                        corner=False)
        g.fig.suptitle(f'Pairwise Feature Relationships - {dataset_name}', 
                      y=1.02, fontsize=16, fontweight='bold')
        
        plt.savefig(f"{PLOTS_DIR}/pairwise_relationships_{dataset_name.lower()}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved plot: pairwise_relationships_{dataset_name.lower()}.png")
    else:
        print("Not enough data for pairwise analysis.")


def analyze_data_quality(df, dataset_name):
    """Overall data quality assessment"""
    print_subheader(f"12. DATA QUALITY SUMMARY - {dataset_name}")
    
    total_rows = len(df)
    total_cells = df.shape[0] * df.shape[1]
    
    # Missing data
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    # Complete rows
    complete_rows = df.dropna().shape[0]
    complete_rows_pct = (complete_rows / total_rows) * 100
    
    # Duplicates
    cols_to_ignore = ['subject_id', 'stay_id']
    df_check = df.drop(columns=[c for c in cols_to_ignore if c in df.columns], errors='ignore')
    duplicates = df_check.duplicated().sum()
    duplicates_pct = (duplicates / total_rows) * 100
    
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"Total Cells: {total_cells:,}")
    print(f"\nMissing Cells: {missing_cells:,} ({missing_pct:.2f}%)")
    print(f"Complete Rows (no missing): {complete_rows:,} ({complete_rows_pct:.2f}%)")
    print(f"Duplicate Rows: {duplicates:,} ({duplicates_pct:.2f}%)")
    
    # Column-wise completeness
    print("\n\nColumn Completeness:")
    print(f"{'Column':<20} {'Non-Null':>12} {'Completeness':>15}")
    print("-" * 50)
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        completeness = (non_null / total_rows) * 100
        print(f"{col:<20} {non_null:>12,} {completeness:>14.2f}%")
    
    # Quality score
    quality_score = (1 - missing_pct/100) * 100
    print(f"\n\nOverall Data Quality Score: {quality_score:.2f}/100")
    print("(Based on completeness, higher is better)")


def generate_summary_statistics(train_df, valid_df, test_df):
    """Generate comprehensive summary statistics"""
    print_subheader("13. COMPREHENSIVE SUMMARY STATISTICS")
    
    print("Dataset Sizes:")
    print(f"  Training:   {len(train_df):>8,} rows ({len(train_df)/(len(train_df)+len(valid_df)+len(test_df))*100:.1f}%)")
    print(f"  Validation: {len(valid_df):>8,} rows ({len(valid_df)/(len(train_df)+len(valid_df)+len(test_df))*100:.1f}%)")
    print(f"  Test:       {len(test_df):>8,} rows ({len(test_df)/(len(train_df)+len(valid_df)+len(test_df))*100:.1f}%)")
    print(f"  Total:      {len(train_df)+len(valid_df)+len(test_df):>8,} rows")
    
    # Memory usage
    train_mem = train_df.memory_usage(deep=True).sum() / 1024**2
    valid_mem = valid_df.memory_usage(deep=True).sum() / 1024**2
    test_mem = test_df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\nMemory Usage:")
    print(f"  Training:   {train_mem:>8.2f} MB")
    print(f"  Validation: {valid_mem:>8.2f} MB")
    print(f"  Test:       {test_mem:>8.2f} MB")
    print(f"  Total:      {train_mem+valid_mem+test_mem:>8.2f} MB")
    
    # Feature types
    print(f"\nFeature Types:")
    print(f"  Numerical:  {len(train_df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical/Text: {len(train_df.select_dtypes(exclude=[np.number]).columns)}")
    print(f"  Total:      {train_df.shape[1]}")


def create_comprehensive_summary_plot(train_df):
    """Create a comprehensive summary visualization"""
    print_subheader("14. GENERATING COMPREHENSIVE SUMMARY VISUALIZATION")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    acuity_counts = train_df['acuity'].value_counts().sort_index()
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    bars_class = ax1.bar(acuity_counts.index, acuity_counts.values, color=colors[:len(acuity_counts)], 
           edgecolor='black', alpha=0.8)
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.set_xlabel('Acuity Level')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars_class:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Missing values heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    missing_matrix = train_df.isnull().astype(int)
    if missing_matrix.sum().sum() > 0:
        sample = missing_matrix.sample(min(1000, len(missing_matrix)), random_state=42)
        sns.heatmap(sample.T, cbar=True, cmap='YlOrRd', ax=ax2, yticklabels=True)
        ax2.set_title('Missing Values Pattern (Sample)', fontweight='bold')
        ax2.set_xlabel('Sample Index')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        ax2.set_title('Missing Values Pattern', fontweight='bold')
    
    # 3. Data completeness
    ax3 = fig.add_subplot(gs[0, 2])
    completeness = (train_df.notna().sum() / len(train_df) * 100).sort_values()
    bars_comp = ax3.barh(range(len(completeness)), completeness.values, color='teal', alpha=0.7)
    ax3.set_yticks(range(len(completeness)))
    ax3.set_yticklabels(completeness.index, fontsize=9)
    ax3.set_xlabel('Completeness (%)')
    ax3.set_title('Feature Completeness', fontweight='bold')
    ax3.axvline(95, color='red', linestyle='--', label='95% threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for bar, val in zip(bars_comp, completeness.values):
        ax3.text(val, bar.get_y() + bar.get_height()/2, f' {val:.1f}%', 
                va='center', fontsize=8, fontweight='bold')
    
    # 4-6. Key feature distributions
    numerical_cols = ['heartrate', 'o2sat', 'temperature']
    for idx, col in enumerate(numerical_cols):
        ax = fig.add_subplot(gs[1, idx])
        # Convert to numeric to handle any non-numeric values
        data = pd.to_numeric(train_df[col], errors='coerce').dropna()
        ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{col.capitalize()} Distribution', fontweight='bold')
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add stats
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, alpha=0.7)
        # Add text box with statistics
        stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
    
    # 7. Heart rate by acuity
    ax7 = fig.add_subplot(gs[2, 0])
    data_list = [pd.to_numeric(train_df[train_df['acuity'] == a]['heartrate'], errors='coerce').dropna() 
                for a in sorted(train_df['acuity'].dropna().unique())]
    labels = [f'A{int(a)}' for a in sorted(train_df['acuity'].dropna().unique())]
    bp = ax7.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Add median labels positioned above boxes
    medians = [np.median(d) for d in data_list]
    max_vals = [d.max() for d in data_list]
    for i, (pos, med, max_val) in enumerate(zip(range(1, len(medians) + 1), medians, max_vals)):
        y_pos = max_val + (max_val - med) * 0.03
        ax7.text(pos, y_pos, f'{med:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    ax7.set_title('Heart Rate by Acuity', fontweight='bold')
    ax7.set_ylabel('Heart Rate')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. O2Sat by acuity
    ax8 = fig.add_subplot(gs[2, 1])
    data_list = [pd.to_numeric(train_df[train_df['acuity'] == a]['o2sat'], errors='coerce').dropna() 
                for a in sorted(train_df['acuity'].dropna().unique())]
    bp = ax8.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Add median labels positioned above boxes
    medians = [np.median(d) for d in data_list]
    max_vals = [d.max() for d in data_list]
    for i, (pos, med, max_val) in enumerate(zip(range(1, len(medians) + 1), medians, max_vals)):
        y_pos = max_val + (max_val - med) * 0.03
        ax8.text(pos, y_pos, f'{med:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    ax8.set_title('O2 Saturation by Acuity', fontweight='bold')
    ax8.set_ylabel('O2 Saturation (%)')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Temperature by acuity
    ax9 = fig.add_subplot(gs[2, 2])
    data_list = [pd.to_numeric(train_df[train_df['acuity'] == a]['temperature'], errors='coerce').dropna() 
                for a in sorted(train_df['acuity'].dropna().unique())]
    bp = ax9.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Add median labels positioned above boxes
    medians = [np.median(d) for d in data_list]
    max_vals = [d.max() for d in data_list]
    for i, (pos, med, max_val) in enumerate(zip(range(1, len(medians) + 1), medians, max_vals)):
        y_pos = max_val + (max_val - med) * 0.03
        ax9.text(pos, y_pos, f'{med:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    ax9.set_title('Temperature by Acuity', fontweight='bold')
    ax9.set_ylabel('Temperature (°F)')
    ax9.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Comprehensive EDA Summary - Training Data', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(f"{PLOTS_DIR}/comprehensive_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot: comprehensive_summary.png")


def main():
    """Main EDA execution"""
    
    # Redirect output to both console and file
    tee = Tee(REPORT_FILE)
    sys.stdout = tee
    
    try:
        print_header("COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
        print_header("PATIENT TRIAGE PREDICTION DATASET")
        
        print(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report file: analysis/eda_report.txt")
        print(f"Plots directory: analysis/eda_plots/")
        
        # Load data
        train_df, valid_df, test_df = load_datasets()
        
        # Comprehensive analysis on training data
        analyze_basic_info(train_df, "TRAIN")
        analyze_missing_values(train_df, "TRAIN")
        analyze_duplicates(train_df, "TRAIN")
        analyze_target_distribution(train_df, valid_df, test_df)
        analyze_numerical_features(train_df, "TRAIN")
        analyze_outliers(train_df, "TRAIN")
        analyze_correlation(train_df, "TRAIN")
        analyze_features_by_target(train_df, "TRAIN")
        analyze_text_features(train_df, "TRAIN")
        analyze_text_by_acuity(train_df, "TRAIN")
        analyze_pairwise_relationships(train_df, "TRAIN")
        analyze_data_quality(train_df, "TRAIN")
        
        # Summary across all splits
        generate_summary_statistics(train_df, valid_df, test_df)
        
        # Create comprehensive summary plot
        create_comprehensive_summary_plot(train_df)
        
        # Quick analysis on validation and test sets
        print_header("VALIDATION SET - QUICK ANALYSIS")
        analyze_missing_values(valid_df, "VALID")
        analyze_duplicates(valid_df, "VALID")
        analyze_data_quality(valid_df, "VALID")
        
        print_header("TEST SET - QUICK ANALYSIS")
        analyze_missing_values(test_df, "TEST")
        analyze_duplicates(test_df, "TEST")
        analyze_data_quality(test_df, "TEST")
        
        # Final summary
        print_header("EDA COMPLETE!")
        print(f"\n✓ All visualizations saved to: analysis/eda_plots/")
        print(f"✓ Full report saved to: analysis/eda_report.txt")
        print(f"\nTotal plots generated: {len([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS SUMMARY")
        print("="*80)
        
        print("\n1. Dataset Size:")
        print(f"   - Training: {len(train_df):,} samples")
        print(f"   - Validation: {len(valid_df):,} samples")
        print(f"   - Test: {len(test_df):,} samples")
        
        print("\n2. Class Imbalance:")
        train_counts = train_df['acuity'].value_counts().sort_index()
        max_class = train_counts.max()
        for acuity, count in train_counts.items():
            ratio = max_class / count
            print(f"   - Acuity {int(acuity)}: {count:,} samples (1:{ratio:.1f} ratio)")
        
        print("\n3. Missing Data:")
        missing_cols = train_df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                pct = count / len(train_df) * 100
                print(f"   - {col}: {count:,} ({pct:.1f}%)")
        else:
            print("   - No missing values!")
        
        print("\n4. Data Quality:")
        complete_rows = train_df.dropna().shape[0]
        print(f"   - Complete rows: {complete_rows:,} ({complete_rows/len(train_df)*100:.1f}%)")
        
        duplicates = train_df.drop(columns=['subject_id', 'stay_id'], errors='ignore').duplicated().sum()
        print(f"   - Duplicate rows: {duplicates:,} ({duplicates/len(train_df)*100:.1f}%)")
        
        print("\n" + "="*80)
        print("EDA process completed successfully!")
        print("="*80)
        
    finally:
        sys.stdout = tee.stdout
        tee.close()
        
    print(f"\n✓ EDA complete! Check 'analysis/eda_report.txt' for full report.")
    print(f"✓ Check 'analysis/eda_plots/' for all visualizations.")


if __name__ == "__main__":
    main()
