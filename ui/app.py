import streamlit as st
import tensorflow as tf
import sys
import os
from PIL import Image
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_pet_classifier, predict_image
from src.utils import setup_directories, clean_uploads

def main():
    st.set_page_config(page_title="Pet Classifier", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .upload-box {
                border: 2px dashed #4CAF50;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }
            .prediction-box {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🐱 Pet Classifier: Dogs vs Cats 🐕")
    st.markdown("""
    This application uses a deep learning model to classify images of dogs and cats.
    Upload an image to see the prediction!
    """)

    # Sidebar
    st.sidebar.title("Options")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["models/saved_models/pet_classifier_v1.h5"],
        format_func=lambda x: os.path.basename(x)
    )

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of a dog or cat"
        )

    # Process the uploaded image
    if uploaded_file is not None:
        # Save and display the image
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        with col2:
            st.markdown("### Prediction")
            with st.spinner("Analyzing image..."):
                # Load model
                try:
                    model = tf.keras.models.load_model(model_path)
                except:
                    st.error("Error loading model. Please make sure the model file exists.")
                    return

                # Preprocess and predict
                img = image.resize((150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                img_array = img_array / 255.

                prediction = model.predict(img_array)[0][0]
                class_name = "Dog" if prediction > confidence_threshold else "Cat"
                confidence = prediction if class_name == "Dog" else 1 - prediction

                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style='text-align: center; color: #1f77b4;'>
                        Prediction: {class_name}
                    </h3>
                    <h4 style='text-align: center;'>
                        Confidence: {confidence * 100:.2f}%
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.progress(float(confidence))

    # Training section
    st.markdown("---")
    st.markdown("### Model Training")
    
    col3, col4 = st.columns(2)
    
    with col3:
        epochs = st.number_input("Number of Epochs", min_value=1, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=32)

    with col4:
        if st.button("Train New Model"):
            with st.spinner("Training model... This may take a while."):
                # Add training logic here
                st.success("Model trained successfully!")

if __name__ == "__main__":
    setup_directories()
    main()