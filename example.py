"""
Simple Example: Medical Image Classification

This script shows how to use the simplified model framework to train and evaluate 
models on a pneumonia X-ray classification task.
"""

from utils.model_manager import ModelManager

def main():
    # Create a model manager
    # Change the data directory to point to your dataset path
    manager = ModelManager(data_dir="data", img_size=(150, 150), batch_size=32)
    
    # Visualize data distribution
    print("Visualizing data distribution...")
    manager.visualize_data_distribution()
    
    # Initialize and train the custom CNN model
    print("\nTraining Custom CNN model...")
    manager.initialize_model("custom_cnn")
    manager.train_model("custom_cnn", epochs=10)
    
    # Evaluate the model and show results
    print("\nEvaluating Custom CNN model...")
    accuracy, loss, report = manager.evaluate_model("custom_cnn")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize model performance
    model = manager.get_model("custom_cnn")
    model.plot_training_history()
    model.plot_confusion_matrix()
    
    # Example of using DenseNet121 (a pre-trained model)
    print("\nTraining DenseNet121 model...")
    manager.initialize_model("densenet121")
    manager.train_model("densenet121", epochs=5)
    
    # Compare the two models
    print("\nComparing models...")
    manager.compare_models(["custom_cnn", "densenet121"])
    
    print("\nFor more options, try using train_models.py with command-line arguments.")

if __name__ == "__main__":
    main() 