import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Get the absolute path to the models directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Set page config
st.set_page_config(
    page_title="Pet Classifier: Dogs vs Cats",
    page_icon="🐱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model(model_name='final_model.keras'):
    """Load the trained model"""
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize image
    img = image.resize((160, 160))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(image, model):
    """Predict whether the image is a cat or dog"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get class label (0 = cat, 1 = dog)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100
        
        # Return result
        class_label = 'Dog' if class_idx == 1 else 'Cat'
        return class_label, confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def main():
    # Title
    st.title("🐱 Pet Classifier: Dogs vs Cats 🐶")
    st.write("This application uses a deep learning model to classify images of dogs and cats. Upload an image and see the prediction!")
    
    # Sidebar
    with st.sidebar:
        st.header("Options")
        
        # Model selection
        st.subheader("Select Model")
        model_version = st.selectbox(
            "Choose model version",
            ["pet_classifier_v1.h5", "final_model.keras"],
            index=1
        )
        
        # Confidence threshold
        st.subheader("Confidence Threshold")
        confidence_threshold = st.slider(
            "Minimum confidence required",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    # Load model
    model = load_model(model_version)
    
    if model is None:
        st.error("Please make sure the model file exists in the models directory.")
        return
    
    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a cat or dog image (JPG, JPEG, PNG)"
    )
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image..."):
                class_label, confidence = predict_image(image, model)
                
                if class_label and confidence:
                    # Display prediction only if confidence is above threshold
                    if confidence/100 >= confidence_threshold:
                        st.success(f"Prediction: {class_label}")
                        st.progress(float(confidence/100))  # Convert to Python float
                        st.info(f"Confidence: {confidence:.2f}%")
                    else:
                        st.warning("Confidence too low to make a prediction")
                        st.info(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
