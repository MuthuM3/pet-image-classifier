import os
import shutil
from datetime import datetime

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'data/train/dogs',
        'data/train/cats',
        'data/test/dogs',
        'data/test/cats',
        'data/uploads',
        'models/saved_models'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def clean_uploads():
    """Clean the uploads directory"""
    if os.path.exists('data/uploads'):
        shutil.rmtree('data/uploads')
        os.makedirs('data/uploads')