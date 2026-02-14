"""
Save SHAP Background Data

This script samples a subset of the training data to use as background
for SHAP explanations in the prediction service.

Run this ONCE after training the model to prepare SHAP background data.
"""

import pandas as pd
import os

def merge_classes(y):
    """Merge classes 4, 5 -> 3 for 3-class problem"""
    return y.replace({4: 3, 5: 3})

def save_shap_background(n_samples=500):
    """
    Sample training data for SHAP background
    
    Args:
        n_samples: Number of background samples (default 500)
    """
    print("=" * 70)
    print("SAVING SHAP BACKGROUND DATA")
    print("=" * 70)
    
    # Load processed training data
    train_path = "../ml_model/ml_processed_data/ml_processed_train.csv"
    output_path = "xai_outputs/shap_background.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ Error: Training data not found at {train_path}")
        print("   Please run ml_preprocess.py first.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    print(f"\nLoading training data from: {train_path}")
    print("(This may take a moment for large files...)")
    
    # Read only needed rows using nrows parameter for speed
    # We'll read more than needed and sample to get diverse examples
    chunk_size = min(n_samples * 10, 10000)
    train_df = pd.read_csv(train_path, nrows=chunk_size)
    
    print(f"Loaded {len(train_df)} rows")
    
    # Remove target column if present and merge classes
    if 'acuity' in train_df.columns:
        # Merge classes 4, 5 -> 3 to match model training
        train_df['acuity'] = merge_classes(train_df['acuity'])
        features = train_df.drop(columns=['acuity'])
        target = train_df['acuity']
        print(f"Target distribution (after merging classes 4,5->3): {target.value_counts().to_dict()}")
    else:
        features = train_df
    
    # Sample background data
    if len(features) > n_samples:
        # Stratified sampling if target is available
        if 'acuity' in train_df.columns:
            print(f"\nPerforming stratified sampling to get {n_samples} diverse samples...")
            background_with_target = train_df.groupby('acuity', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_samples // 3), random_state=42)
            ).head(n_samples)
            background = background_with_target.drop(columns=['acuity'])
        else:
            print(f"\nRandomly sampling {n_samples} samples...")
            background = features.sample(n=n_samples, random_state=42)
    else:
        print(f"\nUsing all {len(features)} available samples...")
        background = features
    
    # Save background data
    background.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved SHAP background data to: {output_path}")
    print(f"  Shape: {background.shape}")
    print(f"  Features: {background.shape[1]}")
    print(f"  Samples: {background.shape[0]}")
    print("\n" + "=" * 70)
    print("SHAP background data is ready!")
    print("The prediction service can now generate SHAP explanations.")
    print("=" * 70)


if __name__ == "__main__":
    # Increase to 500-1000 for more accurate SHAP values (but slower)
    save_shap_background(n_samples=500)
