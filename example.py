"""
Simplified Example: Medical Image Classification

This script shows how to use the simplified model framework to train and evaluate 
a model on a pneumonia X-ray classification task.
"""

from utils.model_manager import ModelManager

def main():
    # Create a model manager
    # Change the data directory to point to your dataset path
    manager = ModelManager(data_dir="data", img_size=(150, 150), batch_size=32)
    
    # Initialize the models you want to work with
    manager.initialize_model("custom_cnn")
    
    # Train the model
    # Adjust epochs as needed
    print("Training model...")
    manager.train_model("custom_cnn", epochs=10)
    
    # Evaluate the model
    print("Evaluating model...")
    accuracy, loss, report = manager.evaluate_model("custom_cnn")
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Example of using another model
    print("\nInitializing and training DenseNet121...")
    manager.initialize_model("densenet121")
    manager.train_model("densenet121", epochs=5)
    
    # You can compare results manually
    print("\nDenseNet121 Evaluation:")
    accuracy, loss, report = manager.evaluate_model("densenet121")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

if __name__ == "__main__":
    main() 