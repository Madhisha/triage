from datasets import load_dataset
import pandas as pd

dataset_name = "dischargesum/triage"
print(f"Downloading dataset: {dataset_name}")
ds = load_dataset(dataset_name)

# Save to CSV
for split, dataset in ds.items():
    output_file = f"raw_data/triage_{split}.csv"
    dataset.to_csv(output_file)
    print(f"Saved {split} split to {output_file}")