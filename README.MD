# Pet Classifier

A deep learning application that classifies images of cats and dogs using TensorFlow and Flask.

## Features

- Image upload functionality
- Real-time classification of cat and dog images
- Web-based user interface
- Pre-trained model for quick predictions

## Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MuthuM3/pet-classifier.git
   cd pet-classifier
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
pet_classifier/
├── data/
│   ├── uploads/        # Folder for uploaded images
│   ├── train/         # Training dataset (not included in repo)
│   └── small_train/   # Small training dataset (not included in repo)
├── models/
│   └── saved_models/  # Trained model files (not included in repo)
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
├── app.py            # Main Flask application
└── requirements.txt  # Project dependencies
```

## Dataset Setup

The training datasets are not included in the repository due to size constraints. To set up the training data:

1. Create directories:
   - `data/train/cats`
   - `data/train/dogs`
   - `data/small_train/cats`
   - `data/small_train/dogs`

2. Add your cat and dog images to respective directories
   - Images should be in common formats (jpg, png)
   - Ensure proper categorization in respective folders

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Upload an image using the web interface
2. The application will classify the image as either a cat or a dog
3. Results will be displayed on the screen

## Model Training

The model has been pre-trained on a dataset of cat and dog images. If you want to retrain the model:

1. Set up the training data as described above
2. Run the training script (if included in your repository)
3. The new model will be saved in the `models` directory

## Contributing

Feel free to fork the repository and submit pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).