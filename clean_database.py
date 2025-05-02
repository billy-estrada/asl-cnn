import os

DATASET_PATH = 'dataset_split/val'  # change if needed

deleted_files = 0
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith('.jpg'):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            deleted_files += 1

print(f"Deleted {deleted_files} .jpg files from {DATASET_PATH}")
