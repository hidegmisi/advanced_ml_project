#!/usr/bin/env python
# Example script for using the pneumonia detection models
import os
import argparse
from utils.model_manager import ModelManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pneumonia Detection Models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--model', type=str, default='all',
                        choices=['densenet121', 'custom_cnn', 'capsnet', 'all'],
                        help='Model to use')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--compare', action='store_true',
                        help='Compare models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to load weights from')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create directories for saving weights and histories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('history', exist_ok=True)
    
    # Initialize the model manager
    print("Initializing model manager...")
    manager = ModelManager(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Determine which models to use
    if args.model == 'all':
        models = ['densenet121', 'custom_cnn', 'capsnet']
    else:
        models = [args.model]
    
    # Initialize models
    for model_name in models:
        print(f"Initializing {model_name}...")
        manager.initialize_model(model_name)
    
    # Train models if requested
    if args.train:
        for model_name in models:
            print(f"Training {model_name}...")
            history = manager.train_model(model_name, epochs=args.epochs)
            
            # Save training history
            history_path = f"history/{model_name}_history.json"
            manager.models[model_name].save_training_history(history_path)
            print(f"Training history saved to {history_path}")
            
            # Plot training history
            manager.models[model_name].plot_training_history()
    
    # Evaluate models if requested
    if args.evaluate:
        print("\nModel Evaluation Results:")
        print("--------------------------")
        
        for model_name in models:
            # Load weights if provided
            if args.weights_path:
                weights_file = os.path.join(args.weights_path, f"{model_name}_weights.h5")
                if os.path.exists(weights_file):
                    print(f"Loading weights for {model_name} from {weights_file}")
                    manager.load_model_weights(model_name, weights_file)
            
            # Evaluate model
            print(f"\nEvaluating {model_name}...")
            accuracy, loss, report = manager.evaluate_model(model_name)
            
            # Print results
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
            print(f"Classification Report:\n{report}")
            
            # Plot confusion matrix
            manager.models[model_name].plot_confusion_matrix()
    
    # Compare models if requested
    if args.compare and len(models) > 1:
        print("\nComparing Models...")
        
        # Compare model performance
        results = manager.compare_models(model_names=models)
        
        # Compare training histories
        history_paths = [f"history/{model_name}_history.json" for model_name in models]
        history_exists = all(os.path.exists(path) for path in history_paths)
        
        if history_exists:
            print("Comparing training histories...")
            manager.compare_training_history(model_names=models, history_paths=history_paths)
        else:
            print("Training histories not available for all models. Skipping history comparison.")

if __name__ == "__main__":
    main()
    print("\nExample script completed.")
    print("Usage examples:")
    print("  Train all models: python example.py --train")
    print("  Train specific model: python example.py --model custom_cnn --train")
    print("  Evaluate model: python example.py --model densenet121 --evaluate")
    print("  Compare all models: python example.py --compare")
    print("  Train and evaluate: python example.py --train --evaluate") 