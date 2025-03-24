import os
import sys

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
        
        return self.models[model_name]
    
    def train_model(self, model_name, epochs=10):
        """Train a specific model."""
        model = self.get_model(model_name)
        
        print(f"Training {model_name} for {epochs} epochs...")
        history = model.train(epochs=epochs)
        
        return history
    
    def evaluate_model(self, model_name):
        """Evaluate a specific model on the test set."""
        model = self.get_model(model_name)
        return model.evaluate()
    
    def get_model(self, model_name):
        """Get a model by name, initializing it if necessary."""
        model_name = model_name.lower()
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(self.models.keys())}")
            
        if self.models[model_name] is None:
            self.initialize_model(model_name)
            
        return self.models[model_name]

# Example usage
if __name__ == "__main__":
    # Create model manager
    manager = ModelManager()
    
    # Initialize and train a model
    manager.initialize_model("custom_cnn")
    manager.train_model("custom_cnn", epochs=5)
    
    # Evaluate the model
    test_acc, test_loss, report = manager.evaluate_model("custom_cnn")
    print(f"Test accuracy: {test_acc}") 