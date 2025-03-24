import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

class BaseModel:
    def __init__(self, img_size=(150, 150), batch_size=32):
        """Initialize the base model with common parameters."""
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.model = None
        self.history = None
        self.model_name = "base_model"
        
        # Create directories for saving
        os.makedirs('weights', exist_ok=True)
        os.makedirs('history', exist_ok=True)
    
    def setup_data_generators(self, train_dir="data/train", val_dir="data/val", test_dir="data/test"):
        """Set up data generators for training, validation, and testing."""
        # Simple data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True
        )

        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Load images from directories
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode="binary"
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
    
    def build_model(self):
        """Build the model. Must be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, epochs=10, save_weights=True):
        """Train the model with early stopping and optional weight saving."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Add weight saving if requested
        if save_weights:
            callbacks.append(
                ModelCheckpoint(
                    filepath=f"weights/{self.model_name}_best.weights.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=True
                )
            )
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
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
        
        return test_acc, test_loss, report
        
    def plot_training_history(self):
        """Plot training and validation accuracy/loss."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the model predictions."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Get true labels and predictions
        true_labels = self.test_generator.classes
        predictions = self.model.predict(self.test_generator)
        predicted_labels = (predictions > 0.5).astype(int).reshape(-1)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        class_names = list(self.test_generator.class_indices.keys())
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    
    def show_data_distribution(self, train_dir="data/train", val_dir="data/val", test_dir="data/test"):
        """Visualize class distribution in the dataset."""
        def count_images(directory):
            if not os.path.exists(directory):
                return 0, 0
                
            normal_dir = os.path.join(directory, 'NORMAL')
            pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
            
            normal_count = len([f for f in os.listdir(normal_dir) if not f.startswith('.')]) if os.path.exists(normal_dir) else 0
            pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')]) if os.path.exists(pneumonia_dir) else 0
            
            return normal_count, pneumonia_count
            
        # Count images in each set
        train_normal, train_pneumonia = count_images(train_dir)
        val_normal, val_pneumonia = count_images(val_dir)
        test_normal, test_pneumonia = count_images(test_dir)
        
        # Create data for bar plot
        datasets = ['Train', 'Validation', 'Test']
        normal_counts = [train_normal, val_normal, test_normal]
        pneumonia_counts = [train_pneumonia, val_pneumonia, test_pneumonia]
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, normal_counts, width, label='Normal')
        plt.bar(x + width/2, pneumonia_counts, width, label='Pneumonia')
        
        plt.xlabel('Dataset')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution Across Datasets')
        plt.xticks(x, datasets)
        plt.legend()
        
        # Add counts as text
        for i, v in enumerate(normal_counts):
            plt.text(i - width/2, v + 5, str(v), ha='center')
        for i, v in enumerate(pneumonia_counts):
            plt.text(i + width/2, v + 5, str(v), ha='center')
            
        plt.tight_layout()
        plt.show()
        
        # Print class imbalance summary
        print(f"Training set: {train_normal} normal, {train_pneumonia} pneumonia images")
        print(f"Validation set: {val_normal} normal, {val_pneumonia} pneumonia images")
        print(f"Test set: {test_normal} normal, {test_pneumonia} pneumonia images")
        
        # Calculate class imbalance
        train_ratio = train_pneumonia / train_normal if train_normal > 0 else 0
        print(f"Training set class imbalance: {train_ratio:.2f}:1 (Pneumonia:Normal)")
        
    def save_model(self, filepath=None):
        """Save the model weights to a file."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if filepath is None:
            filepath = f"weights/{self.model_name}_final.weights.h5"
            
        # Save weights
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}") 