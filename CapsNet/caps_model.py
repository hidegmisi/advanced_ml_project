import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CapsNet:
    def __init__(self, img_size=(28, 28), batch_size=32, routings=3):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.routings = routings
        self.model = None
        self.history = None
        self.model_name = "capsnet"
        
        # Create directories for saving weights and history
        os.makedirs('weights', exist_ok=True)
        os.makedirs('history', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Check TensorFlow version and device
        print("TensorFlow version:", tf.__version__)
        print("Using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
        
        # Configure GPU memory growth if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    
    def setup_data_generators(self, train_dir="train", val_dir="val", test_dir="test"):
        """Set up data generators for training, validation, and testing."""
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Validation and test data only need rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create data generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            color_mode='grayscale'  # Use grayscale for CapsNet
        )

        self.val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            color_mode='grayscale'
        )

        self.test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False  # Don't shuffle test data to preserve order for evaluation
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Found {self.test_generator.samples} test images")
    
    def squash(self, vectors, axis=-1):
        """Squashing function for capsule outputs"""
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-8)
        return scale * vectors
    
    def build_model(self):
        """Build a simplified CapsNet-inspired model for binary classification"""
        # Input layer
        input_shape = (*self.IMG_SIZE, 1)  # Grayscale images
        inputs = layers.Input(shape=input_shape, name='input_layer')
        
        # Convolutional feature extraction
        x = layers.Conv2D(64, 3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        
        # Primary capsules (simplified)
        primary_caps = layers.Conv2D(8 * 16, kernel_size=3, strides=1, padding='valid')(x)
        primary_caps_reshaped = layers.Reshape((-1, 8))(primary_caps)
        
        # Apply squashing
        primary_caps_squashed = layers.Lambda(
            self.squash,
            output_shape=lambda x: x,
            name='primary_caps_squashed'
        )(primary_caps_reshaped)
        
        # Capsule feature aggregation using global average pooling
        x = layers.GlobalAveragePooling1D()(primary_caps_squashed)
        
        # Output prediction layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
    
    def train(self, epochs=50, save_best_only=True):
        """Train the model with callbacks for saving weights and early stopping."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Calculate steps per epoch and validation steps
        steps_per_epoch = self.train_generator.samples // self.BATCH_SIZE
        validation_steps = self.val_generator.samples // self.BATCH_SIZE
        
        # Ensure steps are at least 1
        steps_per_epoch = max(1, steps_per_epoch)
        validation_steps = max(1, validation_steps)
        
        # Define callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            # Save best model weights
            ModelCheckpoint(
                f'weights/{self.model_name}_{timestamp}.weights.h5',
                save_best_only=save_best_only,
                save_weights_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            # Reduce learning rate when a metric has stopped improving
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
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
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Generate predictions
        test_steps = np.ceil(self.test_generator.samples / self.BATCH_SIZE).astype(int)
        predictions = self.model.predict(self.test_generator, steps=test_steps)
        
        # For binary classification
        predicted_classes = (predictions > 0.5).astype(int)
        
        # True labels
        true_classes = self.test_generator.classes
        
        # Print classification report
        class_names = list(self.test_generator.class_indices.keys())
        report = classification_report(true_classes, predicted_classes, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes, class_names)
        
        return test_acc, test_loss, report
    
    def plot_confusion_matrix(self, true_classes, predicted_classes, class_names):
        """Plot confusion matrix."""
        cm = confusion_matrix(true_classes, predicted_classes.flatten())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the entire model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = f"models/{self.model_name}_{timestamp}"
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def visualize_features(self, data_batch, num_samples=5):
        """Visualize capsule features"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        try:
            # Get an intermediate model that outputs the capsule features
            feature_model = models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('primary_caps_squashed').output
            )
            
            # Get feature outputs
            feature_outputs = feature_model.predict(data_batch[:num_samples])
            
            # Plot the activations
            plt.figure(figsize=(20, 4))
            for i in range(min(num_samples, len(data_batch))):
                plt.subplot(1, num_samples, i+1)
                # Average activation across all capsules
                avg_activation = np.mean(feature_outputs[i], axis=0)
                plt.bar(range(len(avg_activation)), avg_activation, align='center')
                plt.title(f"Sample {i+1}")
                plt.xlabel('Capsule Dimension')
                plt.ylabel('Average Activation')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error visualizing features: {e}")

# Example usage
if __name__ == "__main__":
    # Create model
    capsnet = CapsNet(img_size=(28, 28))
    
    # Setup data generators
    capsnet.setup_data_generators()
    
    # Build model
    capsnet.build_model()
    
    # Train model
    capsnet.train(epochs=20)
    
    # Evaluate model
    capsnet.evaluate()
    
    # Plot training history
    capsnet.plot_training_history() 