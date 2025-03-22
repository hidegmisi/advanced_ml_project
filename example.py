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
        manager.initialize_model(model_name, weights_path=args.weights_path)
        
        # Train models if requested
        if args.train:
            print(f"Training {model_name}...")
            manager.train_model(model_name, epochs=args.epochs)
            
            history_path = f"history/{model_name}_latest.json"
            print(f"Saving training history to {history_path}")
            manager.models[model_name].save_training_history(history_path)
        
        # Evaluate models if requested
        if args.evaluate:
            print(f"Evaluating {model_name}...")
            manager.evaluate_model(model_name)
    
    # Compare models if requested
    if args.compare and len(models) > 1:
        print("\nComparing Models...")
        manager.compare_models(models)

if __name__ == "__main__":
    main()
    print("\nExample script completed.")
    print("Usage examples:")
    print("  Train all models: python example.py --train")
    print("  Train specific model: python example.py --model custom_cnn --train")
    print("  Evaluate model: python example.py --model densenet121 --evaluate")
    print("  Compare all models: python example.py --compare")
    print("  Train and evaluate: python example.py --train --evaluate") 