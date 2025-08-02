import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_data, preprocess_data
import os

def create_cnn_model():
    """
    Create a CNN model for digit classification (0-9)
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Regularization to prevent overfitting
        layers.Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
    
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, 
                epochs=20, batch_size=32, model_save_path='models/digit_classifier.h5'):
    """
    Train the CNN model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_images, test_labels):
    """
    Evaluate the trained model
    """
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return test_accuracy, test_loss

def main():
    """
    Main training pipeline
    """
    print("=== MNIST Digit Classification CNN ===")
    print("Loading and preprocessing data...")
    
    # Load data
    dataset_path = "./dataset"
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data(dataset_path)
    
    # Preprocess data
    train_images, train_labels, test_images, test_labels = preprocess_data(
        train_images, train_labels, test_images, test_labels
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Image shape: {train_images.shape[1:]}")
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model()
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model, train_images, train_labels, test_images, test_labels,
        epochs=20, batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_images, test_labels)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nTraining completed!")
    print("Model saved as: models/digit_classifier.h5")
    print("Training history plot saved as: training_history.png")

if __name__ == "__main__":
    main()
