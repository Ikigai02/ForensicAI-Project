

import os
import shutil
import random

# --- Section 1: Configuration ---
SOURCE_DIR = "dataset/processed"
BASE_DIR = "dataset/final_split"

# Split Ratios: 80% Training, 10% Validation, 10% Testing
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test ratio will naturally be the remaining 0.1 (10%)

# Create the folder structure
for split in ["train", "val", "test"]:
    for category in ["authentic", "tampered"]:
        os.makedirs(os.path.join(BASE_DIR, split, category), exist_ok=True)

def split_data():
    categories = ["authentic", "tampered"]
    
    print("📂 Starting data split...")

    for category in categories:
        source_path = os.path.join(SOURCE_DIR, category)
        files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle files to ensure the AI doesn't learn based on order
        random.shuffle(files)
        
        # Calculate split points
        total_files = len(files)
        train_end = int(total_files * TRAIN_RATIO)
        val_end = train_end + int(total_files * VAL_RATIO)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Move files to their respective folders
        dataset_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, file_list in dataset_map.items():
            print(f"Moving {len(file_list)} images to {split_name}/{category}...")
            for f in file_list:
                src = os.path.join(source_path, f)
                dst = os.path.join(BASE_DIR, split_name, category, f)
                shutil.copy(src, dst)

    print("-" * 30)
    print(f"✅ Data split complete! Final dataset located in: {BASE_DIR}")

if __name__ == "__main__":
    split_data()
