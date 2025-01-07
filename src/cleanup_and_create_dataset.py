import os
import shutil
from create_small_dataset import create_small_dataset

# Remove existing small_train directory if it exists
small_train_dir = "data/small_train"
if os.path.exists(small_train_dir):
    shutil.rmtree(small_train_dir)
    print(f"Removed existing {small_train_dir}")

# Create new dataset with verified images
create_small_dataset("data/train", small_train_dir, n_samples=1000)
