import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import json
import pickle

class BaseModel:
    def __init__(self, img_size=(224, 224), batch_size=32):
        """Initialize the base model with common parameters.
        
        Args:
            img_size (tuple): Image dimensions (height, width)
            batch_size (int): Batch size for training
        """
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.model = None
        self.history = None
        self.model_name = "base_model"
        
        # Create directories for saving weights and history
        os.makedirs('weights', exist_ok=True)
        os.makedirs('history', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def setup_data_generators(self, train_dir="data/train", val_dir="data/val", test_dir="data/test"):
        """Set up data generators for training, validation, and testing.
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            test_dir (str): Path to test data directory
        """
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

        val_test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation/test

        # Load images from directories
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary"  # Binary classification (Normal vs Pneumonia)
        )

        self.val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary"
        )

        self.test_generator = val_test_datagen.flow_from_directory(
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
        """Build the model. Must be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, epochs=10, save_best_only=True):
        """Train the model with callbacks for saving weights and early stopping.
        
        Args:
            epochs (int): Number of training epochs
            save_best_only (bool): If True, only save best weights
            
        Returns:
            The training history
        """
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
        """Save training history to file.
        
        Args:
            timestamp (str, optional): Timestamp for filename
        """
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
    
    def save_training_history(self, filepath):
        """Save training history to a specified filepath.
        
        Args:
            filepath (str): Path to save the history file
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        # Extract directory and filename
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Save history based on file extension
        if filepath.endswith('.json'):
            history_dict = {key: [float(value) for value in self.history.history[key]] 
                           for key in self.history.history.keys()}
            
            with open(filepath, 'w') as f:
                json.dump(history_dict, f)
        else:
            # Default to pickle if not json
            with open(filepath, 'wb') as f:
                pickle.dump(self.history.history, f)
                
        print(f"Training history saved to {filepath}")
    
    def load_weights(self, weights_path):
        """Load model weights from file.
        
        Args:
            weights_path (str): Path to weights file
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    
    def evaluate(self):
        """Evaluate the model on the test set.
        
        Returns:
            tuple: (test_accuracy, test_loss, classification_report)
        """
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
        """Plot confusion matrix.
        
        Args:
            true_labels (array): Ground truth labels
            predicted_labels (array): Predicted labels
        """
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
    
    def plot_training_history(self, history_path=None):
        """Plot training and validation accuracy/loss.
        
        Args:
            history_path (str, optional): Path to a saved history file (.json or .pkl).
                                         If provided, loads history from file instead of using 
                                         the model's history attribute.
        """
        history_data = None
        
        # If history path provided, load from file
        if history_path:
            if history_path.endswith('.json'):
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
            elif history_path.endswith('.pkl'):
                with open(history_path, 'rb') as f:
                    history_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format for {history_path}. Must be .json or .pkl")
        # Otherwise use model's history
        elif self.history is not None:
            history_data = self.history.history
        else:
            raise ValueError("No training history available. Train the model first or provide a history file path.")
            
        # Plot Accuracy & Loss
        plt.figure(figsize=(12, 4))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history_data['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history_data:
            plt.plot(history_data['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history_data['loss'], label='Train Loss')
        if 'val_loss' in history_data:
            plt.plot(history_data['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')

        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the entire model.
        
        Args:
            filepath (str, optional): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = f"models/{self.model_name}_{timestamp}"
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def display_dataset_distribution(self, train_dir="data/train", val_dir="data/val", test_dir="data/test"):
        """Display dataset distribution.
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory
            test_dir (str): Path to test data directory
        """
        # Function to count images in each class
        def count_images(directory):
            normal_dir = os.path.join(directory, 'NORMAL')
            pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
            
            normal_count = len([f for f in os.listdir(normal_dir) if not f.startswith('.')])
            pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')])
            
            return normal_count, pneumonia_count

        # Count images
        train_normal, train_pneumonia = count_images(train_dir)
        val_normal, val_pneumonia = count_images(val_dir)
        test_normal, test_pneumonia = count_images(test_dir)

        print(f"Training set: {train_normal} normal, {train_pneumonia} pneumonia images")
        print(f"Validation set: {val_normal} normal, {val_pneumonia} pneumonia images")
        print(f"Test set: {test_normal} normal, {test_pneumonia} pneumonia images")

        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Train', 'Validation', 'Test'], 
                    y=[train_normal + train_pneumonia, val_normal + val_pneumonia, test_normal + test_pneumonia])
        plt.title('Dataset Distribution')
        plt.ylabel('Number of Images')
        plt.show()

        # Class balance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Train Normal', 'Train Pneumonia', 'Val Normal', 'Val Pneumonia', 'Test Normal', 'Test Pneumonia'],
                    y=[train_normal, train_pneumonia, val_normal, val_pneumonia, test_normal, test_pneumonia])
        plt.title('Class Distribution')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def display_sample_images(self, train_dir="data/train", num_samples=4):
        """Display sample images from each class.
        
        Args:
            train_dir (str): Path to training data directory
            num_samples (int): Number of samples to display
        """
        import cv2
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(train_dir, class_name)
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

    def display_predictions(self, num_images=8):
        """Visualize model predictions.
        
        Args:
            num_images (int): Number of images to display
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Reset generator to start from the beginning
        self.test_generator.reset()
        
        # Get one batch of test images
        batch_images, batch_labels = next(self.test_generator)
        
        # Make predictions
        predictions = self.model.predict(batch_images)
        pred_classes = (predictions > 0.5).astype(int).flatten()
        
        # Display images with predictions
        plt.figure(figsize=(15, 12))
        for i in range(min(num_images, batch_images.shape[0])):
            plt.subplot(num_images // 4 + 1, 4, i+1)
            plt.imshow(batch_images[i])
            
            true_label = 'PNEUMONIA' if batch_labels[i] == 1 else 'NORMAL'
            pred_label = 'PNEUMONIA' if pred_classes[i] == 1 else 'NORMAL'
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f"True: {true_label}\nPred: {pred_label} ({predictions[i][0]:.2f})", color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show() 