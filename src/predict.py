import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Resize to match model's expected sizing
    img = img.resize((160, 160))  # Updated size to match training
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model_path, image_path):
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess the image
        processed_image = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get the predicted class and confidence
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx] * 100)
        
        # Map class index to label
        class_labels = {0: 'Cat', 1: 'Dog'}
        predicted_class = class_labels[class_idx]
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    model_path = "models/final_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        class_label, confidence = predict_image(model_path, image_path)
        if class_label and confidence:
            print(f"Prediction: {class_label}")
            print(f"Confidence: {confidence:.2f}%")