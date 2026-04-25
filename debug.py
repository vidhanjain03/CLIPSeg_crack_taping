import os
from glob import glob
import config

train_mask_dir = os.path.join(config.WORK_DIR, "train")
mask_paths = glob(os.path.join(train_mask_dir, "*.png"))

print(f"Looking in directory: {train_mask_dir}")
print(f"Does directory exist? {os.path.exists(train_mask_dir)}")
print(f"Number of .png masks found: {len(mask_paths)}")