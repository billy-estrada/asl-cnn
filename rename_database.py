import os
import shutil

OLD_DATASET = 'dataset_old'
NEW_DATASET = 'dataset'

for label in os.listdir(OLD_DATASET):
    old_label_path = os.path.join(OLD_DATASET, label)
    new_label_path = os.path.join(NEW_DATASET, label)

    if not os.path.isdir(old_label_path):
        continue

    os.makedirs(new_label_path, exist_ok=True)

    for fname in os.listdir(old_label_path):
        if not fname.lower().endswith('.jpg'):
            continue

        src_path = os.path.join(old_label_path, fname)
        dst_path = os.path.join(new_label_path, fname)

        # Avoid overwriting by renaming if needed
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
        else:
            # Add a suffix to make the filename unique
            base, ext = os.path.splitext(fname)
            counter = 1
            while True:
                new_fname = f"{base}_{counter}{ext}"
                new_dst_path = os.path.join(new_label_path, new_fname)
                if not os.path.exists(new_dst_path):
                    shutil.copy(src_path, new_dst_path)
                    print(f"Renamed duplicate: {fname} -> {new_fname}")
                    break
                counter += 1
