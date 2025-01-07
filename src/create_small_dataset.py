import os
import shutil
import random
from PIL import Image

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except:
        return False

def create_small_dataset(source_dir, target_dir, n_samples=1000):
    # Create target directories
    os.makedirs(os.path.join(target_dir, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'dogs'), exist_ok=True)
    
    # Process cats
    source_cats_dir = os.path.join(source_dir, 'cats')
    target_cats_dir = os.path.join(target_dir, 'cats')
    cat_images = [img for img in os.listdir(source_cats_dir) 
                 if is_valid_image(os.path.join(source_cats_dir, img))]
    selected_cats = random.sample(cat_images, min(n_samples, len(cat_images)))
    
    print(f"Copying {len(selected_cats)} valid cat images...")
    for img in selected_cats:
        shutil.copy2(
            os.path.join(source_cats_dir, img),
            os.path.join(target_cats_dir, img)
        )
    
    # Process dogs
    source_dogs_dir = os.path.join(source_dir, 'dogs')
    target_dogs_dir = os.path.join(target_dir, 'dogs')
    dog_images = [img for img in os.listdir(source_dogs_dir) 
                 if is_valid_image(os.path.join(source_dogs_dir, img))]
    selected_dogs = random.sample(dog_images, min(n_samples, len(dog_images)))
    
    print(f"Copying {len(selected_dogs)} valid dog images...")
    for img in selected_dogs:
        shutil.copy2(
            os.path.join(source_dogs_dir, img),
            os.path.join(target_dogs_dir, img)
        )
    
    print("Dataset creation complete!")

if __name__ == "__main__":
    source_dir = "data/train"
    target_dir = "data/small_train"
    create_small_dataset(source_dir, target_dir, n_samples=1000)
