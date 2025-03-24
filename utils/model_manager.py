import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob

# Add parent directory to path to import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model classes
from DenseNet121.densenet121_model import DenseNet121Model
from CustomCNN.cnn_model import CustomCNNModel
from CapsNet.capsnet_model import CapsNetModel

class ModelManager:
    def __init__(self, data_dir="data", img_size=(150, 150), batch_size=32):
        """Initialize ModelManager with configuration options."""
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dir = os.path.join(data_dir, "train")
        self.val_dir = os.path.join(data_dir, "val")
        self.test_dir = os.path.join(data_dir, "test")
        
        # Dictionary to store model instances
        self.models = {
            "densenet121": None,
            "custom_cnn": None,
            "capsnet": None
        }
        
        # Create directories for results
        os.makedirs('comparisons', exist_ok=True)
        os.makedirs('weights', exist_ok=True)
    
    def initialize_model(self, model_name, load_best_weights=False):
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
            capsnet_img_size = (128, 128)
            self.models[model_name] = CapsNetModel(img_size=capsnet_img_size, batch_size=self.batch_size)
        
        # Setup data generators
        self.models[model_name].setup_data_generators(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            test_dir=self.test_dir
        )
        
        # Build model
        self.models[model_name].build_model()
        
        # Load best weights if requested and available
        if load_best_weights:
            self._load_best_weights(model_name)
        
        return self.models[model_name]
    
    def _load_best_weights(self, model_name):
        """Helper method to load the best weights for a model if available."""
        # Check for best weights file
        best_weights_path = f"weights/{model_name}_best.weights.h5"
        final_weights_path = f"weights/{model_name}_final.weights.h5"
        
        if os.path.exists(best_weights_path):
            try:
                self.models[model_name].model.load_weights(best_weights_path)
                print(f"Loaded best weights for {model_name} from {best_weights_path}")
                return True
            except:
                print(f"Failed to load weights from {best_weights_path}")
        
        if os.path.exists(final_weights_path):
            try:
                self.models[model_name].model.load_weights(final_weights_path)
                print(f"Loaded final weights for {model_name} from {final_weights_path}")
                return True
            except:
                print(f"Failed to load weights from {final_weights_path}")
        
        # Look for any weights file for this model
        weight_files = glob.glob(f"weights/{model_name}*.weights.h5")
        if weight_files:
            latest_weights = max(weight_files, key=os.path.getmtime)
            try:
                self.models[model_name].model.load_weights(latest_weights)
                print(f"Loaded weights for {model_name} from {latest_weights}")
                return True
            except:
                print(f"Failed to load weights from {latest_weights}")
        
        print(f"No weights found for {model_name}. Using initialized weights.")
        return False
    
    def train_model(self, model_name, epochs=10, save_weights=True):
        """Train a specific model."""
        model = self.get_model(model_name)
        
        print(f"Training {model_name} for {epochs} epochs...")
        history = model.train(epochs=epochs, save_weights=save_weights)
        
        # Save final weights explicitly
        if save_weights:
            model.save_model()
        
        return history
    
    def evaluate_model(self, model_name, load_best_weights=True):
        """Evaluate a specific model on the test set."""
        model = self.get_model(model_name)
        
        # Try to load the best weights before evaluation if not already loaded
        if load_best_weights:
            self._load_best_weights(model_name)
            
        return model.evaluate()
    
    def get_model(self, model_name):
        """Get a model by name, initializing it if necessary."""
        model_name = model_name.lower()
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(self.models.keys())}")
            
        if self.models[model_name] is None:
            self.initialize_model(model_name)
            
        return self.models[model_name]
        
    def compare_models(self, model_names=None):
        """Compare the performance of multiple models."""
        if model_names is None:
            model_names = list(self.models.keys())
            
        # Ensure all models are initialized and trained
        results = {
            "model_name": [],
            "accuracy": [],
            "loss": []
        }
        
        # Evaluate each model and collect results
        for model_name in model_names:
            # Make sure model is initialized
            model = self.get_model(model_name)
            
            # Load best weights for each model before evaluation
            weights_loaded = self._load_best_weights(model_name)
            if not weights_loaded:
                print(f"Warning: No trained weights found for {model_name}. Results may not be meaningful.")
            
            # Evaluate model
            accuracy, loss, _ = model.evaluate()
            
            # Store results
            results["model_name"].append(model_name)
            results["accuracy"].append(accuracy)
            results["loss"].append(loss)
            
        # Create bar plots to compare performance
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        bars = plt.bar(results["model_name"], results["accuracy"])
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 1)
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        # Loss comparison
        plt.subplot(1, 2, 2)
        bars = plt.bar(results["model_name"], results["loss"])
        plt.title("Model Loss Comparison")
        plt.xlabel("Model")
        plt.ylabel("Loss")
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("comparisons/model_comparison.png")
        plt.show()
        
        return results
        
    def visualize_data_distribution(self):
        """Visualize the data distribution across datasets."""
        # Use any model's show_data_distribution method
        # Initialize a model if none exists
        if all(model is None for model in self.models.values()):
            self.initialize_model("custom_cnn")
            
        for model_name, model in self.models.items():
            if model is not None:
                model.show_data_distribution(
                    train_dir=self.train_dir,
                    val_dir=self.val_dir,
                    test_dir=self.test_dir
                )
                break

# Example usage
if __name__ == "__main__":
    # Create model manager
    manager = ModelManager()
    
    # Show data distribution
    manager.visualize_data_distribution()
    
    # Initialize and train models
    manager.initialize_model("custom_cnn")
    manager.initialize_model("densenet121")
    
    manager.train_model("custom_cnn", epochs=5)
    manager.train_model("densenet121", epochs=5)
    
    # Compare models
    manager.compare_models() 