import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import datetime

class DenseNet121Model:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.model = None
        self.history = None
        self.model_name = "densenet121"
        
        # Create directories for saving weights and history
        os.makedirs('weights', exist_ok=True)
        os.makedirs('history', exist_ok=True)
        
    def setup_data_generators(self, train_dir="train", val_dir="val", test_dir="test"):
        """Set up data generators for training, validation, and testing."""
        # Data Augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation/test

        # Load images from directories
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary"  # Binary classification (Normal vs Pneumonia)
        )

        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary"
        )

        self.test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            shuffle=False
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Found {self.test_generator.samples} test images")
        
    def build_model(self):
        """Build the DenseNet121 model with custom top layers."""
        # Load pre-trained DenseNet121 without the top classification layer
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*self.IMG_SIZE, 3))

        # Freeze the base model (so its weights won't be updated during training)
        base_model.trainable = False

        # Add Custom Layers on top
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation="sigmoid")(x)  # Binary classification

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=output)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Display model summary
        self.model.summary()
        
    def train(self, epochs=10, save_best_only=True):
        """Train the model with callbacks for saving weights and early stopping."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Define callbacks
        callbacks = [
            # Save best model weights
            ModelCheckpoint(
                filepath=f"weights/{self.model_name}_{timestamp}.weights.h5",
                monitor='val_accuracy',
                save_best_only=save_best_only,
                save_weights_only=True,
                verbose=1
            ),
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when a metric has stopped improving
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save training history
        self.save_history(timestamp)
        
        return self.history
    
    def save_history(self, timestamp=None):
        """Save training history to file."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
        # Save history as JSON
        history_dict = {key: [float(value) for value in self.history.history[key]] 
                       for key in self.history.history.keys()}
        
        with open(f"history/{self.model_name}_{timestamp}.json", 'w') as f:
            json.dump(history_dict, f)
            
        # Save history as pickle (alternative format)
        with open(f"history/{self.model_name}_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.history.history, f)
            
        print(f"Training history saved to history/{self.model_name}_{timestamp}.json and .pkl")
        
    def load_weights(self, weights_path):
        """Load model weights from file."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
        
    def evaluate(self):
        """Evaluate the model on the test set."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Evaluate on test data
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get true labels and predictions
        true_labels = self.test_generator.classes
        predictions = self.model.predict(self.test_generator)
        predicted_labels = (predictions > 0.5).astype(int).reshape(-1)

        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=list(self.test_generator.class_indices.keys())
        )
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predicted_labels)
        
        return test_acc, test_loss, report
    
    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """Plot confusion matrix."""
        import seaborn as sns
        
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.test_generator.class_indices.keys()),
            yticklabels=list(self.test_generator.class_indices.keys())
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        # Plot Accuracy & Loss
        plt.figure(figsize=(12, 4))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')

        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath=None):
        """Save the entire model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = f"models/{self.model_name}_{timestamp}"
            
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Create model
    densenet = DenseNet121Model()
    
    # Setup data generators
    densenet.setup_data_generators()
    
    # Build model
    densenet.build_model()
    
    # Train model
    densenet.train(epochs=10)
    
    # Evaluate model
    densenet.evaluate()
    
    # Plot training history
    densenet.plot_training_history() 