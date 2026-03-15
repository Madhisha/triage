"""
Class Balancing Script for Triage Model

This script addresses class imbalance in the merged 3-class triage dataset by:
1. Loading the preprocessed ML data
2. Analyzing current class distribution  
3. Applying different balancing strategies (undersampling or SMOTE oversampling)
4. Saving balanced datasets for training, validation, and test sets

Available Strategies:
- undersample_equal: Undersample classes 2 & 3 to match class 1 count (3,858)
- undersample_10k: Undersample classes 2 & 3 to 10,000 samples (default)
- oversample_smote: Oversample class 1 using SMOTE variants to 10,000 samples

SMOTE Types (for oversample_smote strategy):
- regular: Standard SMOTE
- borderline: Borderline SMOTE (focuses on borderline samples)
- svm: SVM SMOTE (uses SVM to find support vectors)
- adasyn: ADASYN (adaptive synthetic sampling)

Note: Only the training set is balanced. Validation and test sets keep original distributions.
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import sys
import argparse

class TeeOutput:
    """Class to write output to both console and file simultaneously"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def print_class_distribution(y, dataset_name="Dataset"):
    """Print detailed class distribution statistics"""
    print(f"\n{'='*70}")
    print(f"{dataset_name} - Class Distribution")
    print(f"{'='*70}")
    
    counts = y.value_counts().sort_index()
    total = len(y)
    
    print(f"Total samples: {total}\n")
    
    for class_label in sorted(y.unique()):
        count = counts[class_label]
        percentage = (count / total) * 100
        print(f"  Class {int(class_label)}: {count:6d} samples ({percentage:5.2f}%)")
    
    # Calculate imbalance ratio
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n  Imbalance Ratio (max/min): {imbalance_ratio:.2f}:1")
    print(f"{'='*70}\n")


def undersample_majority_classes(df, target_col='acuity', sampling_strategy='auto', random_state=42):
    """
    Undersample majority classes to balance the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to balance
    target_col : str
        Name of the target column
    sampling_strategy : str or dict
        - 'auto': Balance all classes to match the minority class
        - 'moderate': Balance to 1.5x the minority class count
        - dict: Custom counts per class {1.0: 10000, 2.0: 10000, 3.0: 10000}
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Balanced dataframe
    """
    print(f"\n{'='*70}")
    print(f"UNDERSAMPLING PROCESS")
    print(f"{'='*70}")
    print(f"Strategy: {sampling_strategy}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get current class counts
    class_counts = y.value_counts().sort_index()
    min_count = class_counts.min()
    
    print(f"\nMinority class count: {min_count}")
    
    # Determine target counts for each class
    if sampling_strategy == 'auto':
        # Balance to minority class
        target_counts = {cls: min_count for cls in class_counts.index}
        print(f"Target: Balance all classes to {min_count} samples")
        
    elif sampling_strategy == 'moderate':
        # Balance to 1.5x minority class (less aggressive)
        target_count = int(min_count * 1.5)
        target_counts = {cls: min(target_count, class_counts[cls]) for cls in class_counts.index}
        print(f"Target: Balance all classes to {target_count} samples (or original if smaller)")
        
    elif isinstance(sampling_strategy, dict):
        # Custom strategy
        target_counts = sampling_strategy
        print(f"Target: Custom counts per class")
        
    else:
        raise ValueError(f"Invalid sampling_strategy: {sampling_strategy}")
    
    # Print target distribution
    print(f"\nTarget class distribution:")
    for cls, count in sorted(target_counts.items()):
        current = class_counts[cls]
        reduction = current - count
        reduction_pct = (reduction / current) * 100
        print(f"  Class {int(cls)}: {current:6d} -> {count:6d} (reducing {reduction:5d} samples, -{reduction_pct:.1f}%)")
    
    # Perform undersampling
    balanced_indices = []
    
    np.random.seed(random_state)
    
    for class_label in sorted(y.unique()):
        # Get indices for this class
        class_indices = y[y == class_label].index.tolist()
        
        # Sample the target number
        target_n = target_counts.get(class_label, len(class_indices))
        
        if target_n < len(class_indices):
            # Undersample
            sampled_indices = np.random.choice(class_indices, size=target_n, replace=False)
            print(f"\n  Class {int(class_label)}: Randomly selected {target_n} samples from {len(class_indices)}")
        else:
            # Keep all samples
            sampled_indices = class_indices
            print(f"\n  Class {int(class_label)}: Kept all {len(class_indices)} samples")
        
        balanced_indices.extend(sampled_indices)
    
    # Create balanced dataframe
    balanced_df = df.loc[balanced_indices].copy()
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"Original size: {len(df)} samples")
    print(f"Balanced size: {len(balanced_df)} samples")
    print(f"Reduction: {len(df) - len(balanced_df)} samples ({((len(df) - len(balanced_df)) / len(df)) * 100:.1f}%)")
    print(f"{'='*70}\n")
    
    return balanced_df


def oversample_minority_class_smote(df, target_col='acuity', minority_class=1.0, 
                                   target_count=10000, smote_type='regular', 
                                   random_state=42):
    """
    Oversample minority class using SMOTE variants
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to balance
    target_col : str
        Name of the target column
    minority_class : float
        The class to oversample
    target_count : int
        Target number of samples for minority class
    smote_type : str
        Type of SMOTE: 'regular', 'borderline', 'svm', 'adasyn'
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Balanced dataframe with oversampled minority class
    """
    print(f"\n{'='*70}")
    print(f"SMOTE OVERSAMPLING PROCESS")
    print(f"{'='*70}")
    print(f"SMOTE Type: {smote_type.upper()}")
    print(f"Target Class: {int(minority_class)}")
    print(f"Target Count: {target_count}")
    
    # Import SMOTE variants
    try:
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
    except ImportError:
        print("\n⚠ ERROR: imbalanced-learn library not found!")
        print("Install it with: pip install imbalanced-learn")
        return df
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get current class counts
    class_counts = y.value_counts().sort_index()
    minority_count = class_counts[minority_class]
    
    print(f"\nCurrent class distribution:")
    for cls in sorted(class_counts.index):
        print(f"  Class {int(cls)}: {class_counts[cls]:6,} samples")
    
    print(f"\nMinority class {int(minority_class)} count: {minority_count} -> {target_count}")
    
    # Define sampling strategy: only oversample the minority class
    sampling_strategy = {minority_class: target_count}
    
    # For other classes, keep their current counts
    for cls in class_counts.index:
        if cls != minority_class:
            sampling_strategy[cls] = class_counts[cls]
    
    # Select SMOTE variant
    try:
        if smote_type == 'regular':
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
            print(f"\nUsing: Regular SMOTE")
        elif smote_type == 'borderline':
            smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
            print(f"\nUsing: Borderline SMOTE (focuses on borderline samples)")
        elif smote_type == 'svm':
            smote = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
            print(f"\nUsing: SVM SMOTE (uses SVM to find support vectors)")
        elif smote_type == 'adasyn':
            smote = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=5)
            print(f"\nUsing: ADASYN (adaptive synthetic sampling)")
        else:
            print(f"\n⚠ Unknown SMOTE type: {smote_type}. Using regular SMOTE.")
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
        
        # Apply SMOTE
        print(f"\nApplying {smote_type.upper()} oversampling...")
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df[target_col] = y_resampled
        
        # Shuffle
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print(f"\n{'='*70}")
        print(f"Original size: {len(df)} samples")
        print(f"After SMOTE: {len(balanced_df)} samples")
        print(f"Synthetic samples generated: {len(balanced_df) - len(df)}")
        print(f"{'='*70}\n")
        
        return balanced_df
        
    except Exception as e:
        print(f"\n⚠ ERROR during SMOTE: {e}")
        print("Returning original dataframe.")
        return df


def balance_datasets(input_dir="ml_processed_data", 
                    output_dir="ml_processed_data/balanced",
                    sampling_strategy='auto',
                    random_state=42):
    """
    Balance all three datasets (train, validation, test)
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the preprocessed datasets
    output_dir : str
        Directory to save balanced datasets
    sampling_strategy : str or dict
        Undersampling strategy ('auto', 'moderate', or custom dict)
    random_state : int
        Random seed for reproducibility
    """
    
    print("\n" + "="*70)
    print("CLASS BALANCING FOR TRIAGE MODEL")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {random_state}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir}")
    
    # Process each dataset
    datasets = {
        'train': 'ml_processed_train.csv',
        'valid': 'ml_processed_valid.csv',
        'test': 'ml_processed_test.csv'
    }
    
    for dataset_name, filename in datasets.items():
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n\n{'#'*70}")
        print(f"PROCESSING: {dataset_name.upper()} SET")
        print(f"{'#'*70}")
        
        # Load dataset
        print(f"\nLoading: {input_path}")
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Show original distribution
        print_class_distribution(df['acuity'], f"BEFORE Balancing - {dataset_name.upper()}")
        
        # Balance the dataset
        balanced_df = undersample_majority_classes(
            df, 
            target_col='acuity',
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        # Show new distribution
        print_class_distribution(balanced_df['acuity'], f"AFTER Balancing - {dataset_name.upper()}")
        
        # Save balanced dataset
        balanced_df.to_csv(output_path, index=False)
        print(f"✓ Saved balanced dataset to: {output_path}")
        
        # Verify save
        verify_df = pd.read_csv(output_path)
        assert len(verify_df) == len(balanced_df), "Save verification failed!"
        print(f"✓ Verified saved file integrity")
    
    print("\n\n" + "="*70)
    print("CLASS BALANCING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nBalanced datasets saved to: {output_dir}/")
    print(f"  - {datasets['train']}")
    print(f"  - {datasets['valid']}")
    print(f"  - {datasets['test']}")
    print("\n" + "="*70 + "\n")


def main(strategy='undersample_10k', smote_type='regular', smote_target_count=10000):
    """
    Main execution function
    
    Parameters:
    -----------
    strategy : str
        Balancing strategy:
        - 'undersample_equal': Undersample classes 2 & 3 to match class 1 count (3,858)
        - 'undersample_10k': Undersample classes 2 & 3 to 10,000 samples
        - 'oversample_smote': Oversample class 1 using SMOTE to 10,000 samples
    smote_type : str
        Type of SMOTE when strategy is 'oversample_smote':
        - 'regular': Regular SMOTE
        - 'borderline': Borderline SMOTE
        - 'svm': SVM SMOTE
        - 'adasyn': ADASYN
    smote_target_count : int
        Target sample count for class 1 when strategy is 'oversample_smote'.
        Supported values: 5000 or 10000.
    """
    
    print("\n" + "="*70)
    print(f"CLASS BALANCING STRATEGY: {strategy.upper()}")
    if strategy == 'oversample_smote':
        print(f"SMOTE TYPE: {smote_type.upper()}")
        print(f"SMOTE TARGET COUNT (CLASS 1): {smote_target_count:,}")
    print("="*70)
    
    # Create output directory
    os.makedirs("ml_processed_data/balanced", exist_ok=True)
    print("\n✓ Created output directory: ml_processed_data/balanced/")
    
    # Train set - Apply balancing based on strategy
    input_path = "ml_processed_data/ml_processed_train.csv"
    output_path = "ml_processed_data/balanced/ml_processed_train.csv"
    
    df_train = pd.read_csv(input_path)
    print_class_distribution(df_train['acuity'], "TRAIN - BEFORE Balancing")
    
    if strategy == 'undersample_equal':
        # Undersample classes 2 & 3 to match class 1 count (3,858)
        print("\nStrategy: Undersample classes 2 & 3 to match class 1 count")
        custom_strategy_train = {1.0: 3858, 2.0: 3858, 3.0: 3858, 4.0: 226, 5.0: 3}
        balanced_train = undersample_majority_classes(df_train, sampling_strategy=custom_strategy_train)
        strategy_desc = "Classes 1, 2, 3: Balanced to EQUAL counts (3,858 samples each)"
        
    elif strategy == 'undersample_10k':
        # Undersample classes 2 & 3 to 10,000
        print("\nStrategy: Undersample classes 2 & 3 to 10,000 samples")
        custom_strategy_train = {1.0: 3858, 2.0: 10000, 3.0: 10000, 4.0: 226, 5.0: 3}
        balanced_train = undersample_majority_classes(df_train, sampling_strategy=custom_strategy_train)
        strategy_desc = "Class 1: 3,858 samples | Classes 2 & 3: 10,000 samples each"
        
    elif strategy == 'oversample_smote':
        # Oversample class 1 using SMOTE to selected target count
        print(f"\nStrategy: Oversample class 1 using {smote_type.upper()} SMOTE to {smote_target_count:,} samples")
        balanced_train = oversample_minority_class_smote(
            df_train, 
            minority_class=1.0, 
            target_count=smote_target_count,
            smote_type=smote_type
        )
        strategy_desc = f"Class 1: Oversampled to {smote_target_count:,} using {smote_type.upper()} SMOTE | Classes 2 & 3: Original"
        
    else:
        print(f"\n⚠ Unknown strategy: {strategy}. Using default undersample_10k.")
        custom_strategy_train = {1.0: 3858, 2.0: 10000, 3.0: 10000, 4.0: 226, 5.0: 3}
        balanced_train = undersample_majority_classes(df_train, sampling_strategy=custom_strategy_train)
        strategy_desc = "Class 1: 3,858 samples | Classes 2 & 3: 10,000 samples each"
    
    print_class_distribution(balanced_train['acuity'], "TRAIN - AFTER Balancing")
    balanced_train.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    # Valid set - Keep original distribution (no balancing)
    input_path = "ml_processed_data/ml_processed_valid.csv"
    output_path = "ml_processed_data/balanced/ml_processed_valid.csv"
    
    df_valid = pd.read_csv(input_path)
    print_class_distribution(df_valid['acuity'], "VALID - Original Distribution (No Balancing Applied)")
    df_valid.to_csv(output_path, index=False)
    print(f"✓ Saved (unbalanced): {output_path}")
    
    # Test set - Keep original distribution (no balancing)
    input_path = "ml_processed_data/ml_processed_test.csv"
    output_path = "ml_processed_data/balanced/ml_processed_test.csv"
    
    df_test = pd.read_csv(input_path)
    print_class_distribution(df_test['acuity'], "TEST - Original Distribution (No Balancing Applied)")
    df_test.to_csv(output_path, index=False)
    print(f"✓ Saved (unbalanced): {output_path}")
    
    # Calculate dataset split percentages
    train_total = len(balanced_train)
    valid_total = len(df_valid)
    test_total = len(df_test)
    grand_total = train_total + valid_total + test_total
    
    train_pct = (train_total / grand_total) * 100
    valid_pct = (valid_total / grand_total) * 100
    test_pct = (test_total / grand_total) * 100
    
    print("\n\n" + "="*70)
    print("CLASS BALANCING COMPLETED!")
    print("="*70)
    print("\nDatasets saved to: ml_processed_data/balanced/")
    print(f"\n  TRAIN SET (Balanced - {strategy}):")
    print(f"    • {strategy_desc}")
    print("    • Classes 4 & 5: Kept at original counts (226 and 3 respectively)")
    print("\n  VALIDATION & TEST SETS (Unbalanced - Original Distribution):")
    print("    • Kept all samples with original class distributions")
    print("    • No undersampling or balancing applied")
    
    print("\n" + "-"*70)
    print("DATASET SPLIT PERCENTAGES")
    print("-"*70)
    print(f"Total samples across all splits: {grand_total:,}")
    print(f"\n  Training Set:   {train_total:6,} samples ({train_pct:5.2f}%) - BALANCED")
    print(f"  Validation Set: {valid_total:6,} samples ({valid_pct:5.2f}%) - ORIGINAL")
    print(f"  Test Set:       {test_total:6,} samples ({test_pct:5.2f}%) - ORIGINAL")
    print("-"*70)
    
    print("="*70 + "\n")


def interactive_menu():
    """
    Display interactive menu for selecting balancing strategy
    
    Returns:
    --------
    tuple: (strategy, smote_type, smote_target_count)
    """
    print("\n" + "="*70)
    print("CLASS BALANCING - INTERACTIVE MENU")
    print("="*70)
    print("\nSelect a balancing strategy:\n")
    print("  1. Undersample classes 2 & 3 to match class 1 (3,858 samples each)")
    print("     → Results in perfectly balanced classes 1, 2, 3")
    print()
    print("  2. Undersample classes 2 & 3 to 10,000 samples")
    print("     → Class 1: 3,858 | Classes 2 & 3: 10,000 each")
    print()
    print("  3. Oversample class 1 using SMOTE to 10,000 samples")
    print("     → Class 1: 10,000 (synthetic) | Classes 2 & 3: original")
    print()
    print("="*70)
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '1':
                return 'undersample_equal', 'regular', 10000
            elif choice == '2':
                return 'undersample_10k', 'regular', 10000
            elif choice == '3':
                # Ask for SMOTE target count
                print("\n" + "-"*70)
                print("Select class 1 target count for regular SMOTE:")
                print("-"*70)
                print("\n  1. 5,000 samples")
                print("  2. 10,000 samples")
                print()
                print("-"*70)

                while True:
                    target_choice = input("\nEnter target option (1 or 2): ").strip()
                    if target_choice == '1':
                        smote_target_count = 5000
                        break
                    elif target_choice == '2':
                        smote_target_count = 10000
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1 or 2.")

                # Ask for SMOTE type
                print("\n" + "-"*70)
                print("Select SMOTE variant:")
                print("-"*70)
                print("\n  1. Regular SMOTE")
                print("     → Standard synthetic minority oversampling")
                print()
                print("  2. Borderline SMOTE")
                print("     → Focuses on borderline samples near decision boundary")
                print()
                print("  3. SVM SMOTE")
                print("     → Uses SVM to identify support vectors for sampling")
                print()
                print("  4. ADASYN")
                print("     → Adaptive synthetic sampling (more samples for harder-to-learn regions)")
                print()
                print("-"*70)
                
                while True:
                    smote_choice = input("\nEnter SMOTE type (1, 2, 3, or 4): ").strip()
                    if smote_choice == '1':
                        return 'oversample_smote', 'regular', smote_target_count
                    elif smote_choice == '2':
                        return 'oversample_smote', 'borderline', smote_target_count
                    elif smote_choice == '3':
                        return 'oversample_smote', 'svm', smote_target_count
                    elif smote_choice == '4':
                        return 'oversample_smote', 'adasyn', smote_target_count
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n⚠ Operation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Balance classes in triage training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with interactive menu (no arguments)
  python balance_classes.py
  
  # OR use command-line arguments:
  
  # Undersample classes 2 & 3 to match class 1 (3,858 samples each)
  python balance_classes.py --strategy undersample_equal
  
  # Undersample classes 2 & 3 to 10,000 samples each
  python balance_classes.py --strategy undersample_10k
  
  # Oversample class 1 using regular SMOTE to 10,000 samples
    python balance_classes.py --strategy oversample_smote --smote regular --smote-target 10000

    # Oversample class 1 using regular SMOTE to 5,000 samples
    python balance_classes.py --strategy oversample_smote --smote regular --smote-target 5000
  
  # Oversample class 1 using Borderline SMOTE
  python balance_classes.py --strategy oversample_smote --smote borderline
  
  # Oversample class 1 using SVM SMOTE
  python balance_classes.py --strategy oversample_smote --smote svm
  
  # Oversample class 1 using ADASYN
  python balance_classes.py --strategy oversample_smote --smote adasyn
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        choices=['undersample_equal', 'undersample_10k', 'oversample_smote'],
        help='Balancing strategy (if not provided, interactive menu will be shown)'
    )
    
    parser.add_argument(
        '--smote',
        type=str,
        default='regular',
        choices=['regular', 'borderline', 'svm', 'adasyn'],
        help='SMOTE type when using oversample_smote strategy (default: regular)'
    )

    parser.add_argument(
        '--smote-target',
        type=int,
        default=10000,
        choices=[5000, 10000],
        help='Target class-1 sample count for oversample_smote (choices: 5000, 10000; default: 10000)'
    )
    
    args = parser.parse_args()
    
    # If no strategy provided, show interactive menu
    if args.strategy is None:
        strategy, smote_type, smote_target_count = interactive_menu()
    else:
        strategy = args.strategy
        smote_type = args.smote
        smote_target_count = args.smote_target
    
    # Create output directory for report
    os.makedirs("../analysis", exist_ok=True)
    
    # Set up output to write to both console and file
    output_file = "../analysis/balance_classes.txt"
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        print("\n" + "="*70)
        print("CLASS BALANCING CONFIGURATION")
        print("="*70)
        print(f"Strategy: {strategy}")
        if strategy == 'oversample_smote':
            print(f"SMOTE Type: {smote_type}")
            print(f"SMOTE Target Count (Class 1): {smote_target_count}")
        print("="*70)
        
        main(strategy=strategy, smote_type=smote_type, smote_target_count=smote_target_count)
        
        print(f"\n{'='*70}")
        print(f"✓ Complete terminal output saved to: analysis/balance_classes.txt")
        print(f"{'='*70}\n")
    finally:
        # Restore original stdout and close file
        sys.stdout = tee.terminal
        tee.close()
        print(f"✓ Log file closed successfully")
