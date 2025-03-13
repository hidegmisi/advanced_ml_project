import os
import sys
import json
import pickle
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Add parent directory to path to import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model classes
from DenseNet121.densenet121_pneumonia import DenseNet121Model
from custom_cnn.pneumonia_cnn_model import CustomCNNModel
from CapsNet.caps_model import CapsNet

class ModelManager:
    def __init__(self, data_dir="data", img_size=(224, 224), batch_size=32):
        """Initialize ModelManager with configuration options."""
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_dir, "train")
        self.val_dir = os.path.join(data_dir, "val")
        self.test_dir = os.path.join(data_dir, "test")
        
        # Create model instances
        self.models = {
            "densenet121": None,
            "custom_cnn": None,
            "capsnet": None
        }
        
        # Create directories for saving comparisons
        os.makedirs('comparisons', exist_ok=True)
    
    def initialize_model(self, model_name):
        """Initialize and configure a specific model."""
        model_name = model_name.lower()
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(self.models.keys())}")
        
        if model_name == "densenet121":
            self.models[model_name] = DenseNet121Model(img_size=self.img_size, batch_size=self.batch_size)
        elif model_name == "custom_cnn":
            self.models[model_name] = CustomCNNModel(img_size=self.img_size, batch_size=self.batch_size)
        elif model_name == "capsnet":
            # CapsNet typically works better with smaller images
            capsnet_img_size = (28, 28)  # Standard size for MNIST-like models
            self.models[model_name] = CapsNet(img_size=capsnet_img_size, batch_size=self.batch_size)
        
        return self.models[model_name]
    
    def initialize_all_models(self):
        """Initialize all available models."""
        for model_name in self.models.keys():
            self.initialize_model(model_name)
    
    def train_model(self, model_name, epochs=10, save_best_only=True):
        """Train a specific model."""
        model_name = model_name.lower()
        
        if model_name not in self.models or self.models[model_name] is None:
            self.initialize_model(model_name)
        
        model = self.models[model_name]
        
        # Setup data generators
        model.setup_data_generators(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir
        )
        
        # Build model
        model.build_model()
        
        # Train model
        history = model.train(epochs=epochs, save_best_only=save_best_only)
        
        return history
    
    def evaluate_model(self, model_name):
        """Evaluate a specific model on the test set."""
        model_name = model_name.lower()
        
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model {model_name} not initialized. Call initialize_model() first.")
        
        model = self.models[model_name]
        
        # Evaluate the model
        test_acc, test_loss, report = model.evaluate()
        
        return test_acc, test_loss, report
    
    def load_model_weights(self, model_name, weights_path):
        """Load weights for a specific model."""
        model_name = model_name.lower()
        
        if model_name not in self.models or self.models[model_name] is None:
            self.initialize_model(model_name)
            
        model = self.models[model_name]
        
        # Setup data generators (needed for some model operations)
        model.setup_data_generators(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir
        )
        
        # Build model
        model.build_model()
        
        # Load weights
        model.load_weights(weights_path)
        
        return model
    
    def compare_models(self, model_names=None, weights_paths=None, save_results=True):
        """Compare multiple models' performance."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        if weights_paths is not None and len(weights_paths) != len(model_names):
            raise ValueError("If weights_paths is provided, it must have the same length as model_names")
        
        results = {
            "model_name": [],
            "accuracy": [],
            "loss": []
        }
        
        # Evaluate each model
        for i, model_name in enumerate(model_names):
            if model_name not in self.models or self.models[model_name] is None:
                self.initialize_model(model_name)
            
            model = self.models[model_name]
            
            # Setup data generators
            model.setup_data_generators(
                train_dir=self.train_dir,
                val_dir=self.val_dir,
                test_dir=self.test_dir
            )
            
            # Build model
            model.build_model()
            
            # Load weights if provided
            if weights_paths is not None:
                model.load_weights(weights_paths[i])
            
            # Evaluate model
            test_acc, test_loss, _ = model.evaluate()
            
            # Store results
            results["model_name"].append(model_name)
            results["accuracy"].append(test_acc)
            results["loss"].append(test_loss)
        
        # Convert to pandas DataFrame
        results_df = pd.DataFrame(results)
        
        # Create comparison plots
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        plt.figure(figsize=(12, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x="model_name", y="accuracy", data=results_df)
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 1.0)
        
        # Loss comparison
        plt.subplot(1, 2, 2)
        sns.barplot(x="model_name", y="loss", data=results_df)
        plt.title("Model Loss Comparison")
        
        plt.tight_layout()
        
        if save_results:
            # Save the comparison plot
            plt.savefig(f"comparisons/model_comparison_{timestamp}.png")
            
            # Save the results data
            results_df.to_csv(f"comparisons/model_comparison_{timestamp}.csv", index=False)
            
            print(f"Comparison results saved to comparisons/ directory")
        
        plt.show()
        
        return results_df
    
    def compare_training_history(self, histories=None, model_names=None, history_paths=None, save_results=True):
        """Compare training histories of multiple models."""
        if histories is None:
            histories = {}
            
            # If neither histories nor history_paths provided, use models' histories
            if history_paths is None:
                for model_name, model in self.models.items():
                    if model is not None and model.history is not None:
                        histories[model_name] = model.history.history
            else:
                # Load histories from provided paths
                if model_names is None or len(model_names) != len(history_paths):
                    raise ValueError("If history_paths is provided, model_names must be provided with the same length")
                
                for i, path in enumerate(history_paths):
                    model_name = model_names[i]
                    
                    if path.endswith('.json'):
                        with open(path, 'r') as f:
                            histories[model_name] = json.load(f)
                    elif path.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            histories[model_name] = pickle.load(f)
                    else:
                        raise ValueError(f"Unsupported file format for {path}. Must be .json or .pkl")
        
        if not histories:
            raise ValueError("No training histories available for comparison")
        
        # Create comparison plots
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        plt.figure(figsize=(15, 10))
        
        # Training accuracy comparison
        plt.subplot(2, 2, 1)
        for model_name, history in histories.items():
            plt.plot(history['accuracy'], label=f"{model_name}")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Validation accuracy comparison
        plt.subplot(2, 2, 2)
        for model_name, history in histories.items():
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label=f"{model_name}")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Training loss comparison
        plt.subplot(2, 2, 3)
        for model_name, history in histories.items():
            plt.plot(history['loss'], label=f"{model_name}")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Validation loss comparison
        plt.subplot(2, 2, 4)
        for model_name, history in histories.items():
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label=f"{model_name}")
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        
        if save_results:
            # Save the comparison plot
            plt.savefig(f"comparisons/training_history_comparison_{timestamp}.png")
            
            print(f"Training history comparison saved to comparisons/ directory")
        
        plt.show()
        
        return histories

# Example usage
if __name__ == "__main__":
    # Create model manager
    manager = ModelManager()
    
    # Initialize models
    manager.initialize_all_models()
    
    # Train models (uncomment to train)
    # manager.train_model("densenet121", epochs=5)
    # manager.train_model("custom_cnn", epochs=5)
    manager.train_model("capsnet", epochs=5)
    
    # Compare models
    # results = manager.compare_models()
    
    print("Model manager initialized successfully. Use the following methods to manage models:")
    print("- initialize_model(model_name): Initialize a specific model")
    print("- train_model(model_name, epochs): Train a specific model")
    print("- evaluate_model(model_name): Evaluate a specific model")
    print("- compare_models(): Compare performance of multiple models")
    print("- compare_training_history(): Compare training histories of multiple models") 