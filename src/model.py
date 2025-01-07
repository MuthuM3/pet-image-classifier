import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

def create_pet_classifier(input_shape=(150, 150, 3)):
    """
    Create a CNN model for binary classification of dogs and cats
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def prepare_data(data_dir, img_height=150, img_width=150, batch_size=32):
    """
    Prepare and augment training and validation data
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, epochs=10):
    """
    Train the model and return training history
    """
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    return history

def predict_image(model, image_path, img_height=150, img_width=150):
    """
    Predict whether an image contains a dog or cat
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.

    prediction = model.predict(img_array)
    return "Dog" if prediction[0] > 0.5 else "Cat"

# Example usage
if __name__ == "__main__":
    # Set up directories
    data_dir = "path/to/dataset"  # Dataset should have 'dogs' and 'cats' subdirectories
    
    # Create and train model
    model = create_pet_classifier()
    train_generator, validation_generator = prepare_data(data_dir)
    history = train_model(model, train_generator, validation_generator)
    
    # Save the trained model
    model.save('pet_classifier.h5')
    
    # Example prediction
    test_image = "path/to/test/image.jpg"
    result = predict_image(model, test_image)
    print(f"The image contains a: {result}")