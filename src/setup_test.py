import os
import shutil

# Create test images directory
test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_images')
os.makedirs(test_dir, exist_ok=True)

# Copy a few images from the training set for testing
train_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'small_train')
cats_dir = os.path.join(train_dir, 'cats')
dogs_dir = os.path.join(train_dir, 'dogs')

# Copy 2 cat images and 2 dog images for testing
for i, (src_dir, animal) in enumerate([(cats_dir, 'cat'), (dogs_dir, 'dog')]):
    if os.path.exists(src_dir):
        for j, img in enumerate(os.listdir(src_dir)[:2]):
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(test_dir, f'test_{animal}_{j+1}{os.path.splitext(img)[1]}')
            shutil.copy2(src_path, dst_path)
            print(f"Copied {dst_path}")

print("\nTest environment setup complete!")
print("You can now test the model using:")
print("python src/predict.py test_images/test_cat_1.jpg")
print("or")
print("python src/predict.py test_images/test_dog_1.jpg")
