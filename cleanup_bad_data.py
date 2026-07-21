import os
import glob

# All folders to search
SEARCH_DIRS = [
    "dataset/raw/tampered",
    "dataset/processed/tampered",
    "dataset/final_split/train/tampered",
    "dataset/final_split/val/tampered",
    "dataset/final_split/test/tampered",
]

total_found = 0
total_deleted = 0

for folder in SEARCH_DIRS:
    if not os.path.exists(folder):
        print(f"Skipping (not found): {folder}")
        continue
    
    print(f"\nSearching: {folder}")
    
    for filename in os.listdir(folder):
        if "visible_tamper" in filename.lower() or "strong_tamper" in filename.lower() or "test_result" in filename.lower():
            filepath = os.path.join(folder, filename)
            total_found += 1
            print(f"  DELETING: {filename}")
            os.remove(filepath)
            total_deleted += 1

print(f"\n{'=' * 50}")
print(f"Done! Found {total_found} files, deleted {total_deleted}.")
print(f"\nNext steps:")
print(f"  1. Run: python 3_split_data.py")
print(f"  2. Run: python 4_train_model_mobilenetv2.py")
print(f"  3. Run: python 7_analyze_model.py")