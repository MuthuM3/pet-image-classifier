import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import os

def create_pet_classifier():
    # Use MobileNetV2 instead of ResNet50V2 (it's lighter and faster)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3)  # Reduced image size
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create a simpler model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(data_dir):
    # Simpler data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Reduced batch size and image size
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(160, 160),  # Reduced image size
        batch_size=16,  # Smaller batch size
        class_mode='sparse',
        subset='training',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(160, 160),  # Reduced image size
        batch_size=16,  # Smaller batch size
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,  # Reduced patience
        restore_best_weights=True
    )
    
    # Train the model with fewer epochs
    history = model.fit(
        train_generator,
        epochs=10,  # Reduced epochs
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
    
    return history

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create and train model
    model = create_pet_classifier()
    train_generator, validation_generator = prepare_data('data/small_train')
    history = train_model(model, train_generator, validation_generator)
    
    # Save the model
    model.save('models/final_model.keras')
    print("Training completed. Model saved to models/final_model.keras")

if __name__ == "__main__":
    main()