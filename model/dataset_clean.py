import os
import glob

root_path = '../dataset'

# Loop through each subfolder in 'dataset'
for subfolder in os.listdir(root_path):
    subfolder_path = os.path.join(root_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        # Get all .jpg files (sorted by filename alphabetically)
        files = sorted(glob.glob(os.path.join(subfolder_path, '*.jpg')))

        # Delete every other file (index 1, 3, 5, ...)
        for i, file_path in enumerate(files):
            if i % 2 == 1:  # Odd index means every other file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
