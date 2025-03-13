import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import pickle
import datetime
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class CustomCNNModel:
    def __init__(self, img_size=(156, 156), batch_size=32):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.model = None
        self.history = None
        self.model_name = "custom_cnn"
        
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
        
        # Create directories for saving weights and history
        os.makedirs('weights', exist_ok=True)
        os.makedirs('history', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def count_images(self, directory):
        """Count the number of images in each class."""
        normal_dir = os.path.join(directory, 'NORMAL')
        pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
        
        normal_count = len([f for f in os.listdir(normal_dir) if not f.startswith('.')])
        pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')])
        
        return normal_count, pneumonia_count
    
    def setup_data_generators(self, train_dir="train", val_dir="val", test_dir="test"):
        """Set up data generators for training, validation, and testing."""
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Count images in each set
        if os.path.exists(train_dir):
            train_normal, train_pneumonia = self.count_images(train_dir)
            print(f"Training set: {train_normal} normal, {train_pneumonia} pneumonia images")
        
        if os.path.exists(val_dir):
            val_normal, val_pneumonia = self.count_images(val_dir)
            print(f"Validation set: {val_normal} normal, {val_pneumonia} pneumonia images")
        
        if os.path.exists(test_dir):
            test_normal, test_pneumonia = self.count_images(test_dir)
            print(f"Test set: {test_normal} normal, {test_pneumonia} pneumonia images")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
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
            class_mode='binary'
        )

        self.val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )

        self.test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='binary',
            shuffle=False  # Don't shuffle test data to preserve order for evaluation
        )
    
    def build_model(self):
        """Build the custom CNN model architecture."""
        self.model = models.Sequential()
        
        self.model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', 
                              input_shape=(*self.IMG_SIZE, 3)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
        
        self.model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
        
        self.model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
        
        self.model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
        
        self.model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=1, activation='sigmoid'))
        
        # Compile the model
        self.model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Display model summary
        self.model.summary()
    
    def train(self, epochs=20, save_best_only=True):
        """Train the model with callbacks for saving weights and early stopping."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Calculate steps per epoch and validation steps
        steps_per_epoch = self.train_generator.samples // self.BATCH_SIZE
        validation_steps = self.val_generator.samples // self.BATCH_SIZE
        
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
                monitor='val_accuracy',
                patience=2,
                verbose=1,
                factor=0.3,
                min_lr=0.000001
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
            
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Generate predictions
        test_steps = np.ceil(self.test_generator.samples / self.BATCH_SIZE).astype(int)
        predictions = self.model.predict(self.test_generator, steps=test_steps)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
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
        cm = confusion_matrix(true_classes, predicted_classes)
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
    
    def display_sample_images(self, class_name, num_samples=4):
        """Display sample images from a class."""
        class_dir = os.path.join(self.train_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if not f.startswith('.')][:num_samples]
        
        plt.figure(figsize=(15, 4))
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"{class_name}: {img.shape}")
            plt.axis('off')
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
    
    def display_predictions(self, num_images=8):
        """Display sample images with predictions."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Reset generator to start from the beginning
        self.test_generator.reset()
        
        # Get a batch of test images and their true labels
        batch = next(self.test_generator)
        images, labels = batch
        
        # Ensure we don't exceed batch size
        num_images = min(num_images, len(images))
        
        # Make predictions
        predictions = self.model.predict(images)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get class names
        class_names = list(self.test_generator.class_indices.keys())
        
        # Display images and predictions
        plt.figure(figsize=(15, 8))
        for i in range(num_images):
            plt.subplot(2, num_images//2, i+1)
            plt.imshow(images[i])
            
            true_class = class_names[int(labels[i])]
            pred_class = class_names[predicted_classes[i]]
            
            color = 'green' if pred_class == true_class else 'red'
            
            plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create model
    cnn_model = CustomCNNModel()
    
    # Setup data generators
    cnn_model.setup_data_generators()
    
    # Build model
    cnn_model.build_model()
    
    # Train model with early stopping
    cnn_model.train(epochs=20)
    
    # Evaluate model
    cnn_model.evaluate()
    
    # Plot training history
    cnn_model.plot_training_history()
    
    # Display some predictions
    cnn_model.display_predictions() 