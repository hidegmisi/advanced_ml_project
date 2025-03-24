#!/usr/bin/env python
"""
Pneumonia X-ray Classification - Model Training and Evaluation

This script provides a command-line interface for training and evaluating 
different deep learning models on the pneumonia X-ray classification task.
"""

import argparse
from utils.model_manager import ModelManager

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate pneumonia classification models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='custom_cnn',
                        choices=['custom_cnn', 'densenet121', 'capsnet', 'all'],
                        help='Model to use (default: custom_cnn)')
    
    # Actions
    parser.add_argument('--train', action='store_true',
                        help='Train the selected model(s)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the selected model(s)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the data distribution')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all trained models')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save model weights during training')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the dataset (default: data)')
    parser.add_argument('--img-size', type=int, default=150,
                        help='Image size for model input (default: 150)')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create model manager
    manager = ModelManager(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Visualize data distribution if requested
    if args.visualize:
        manager.visualize_data_distribution()
    
    # Determine which models to process
    if args.model == 'all':
        models = ['custom_cnn', 'densenet121', 'capsnet']
    else:
        models = [args.model]
    
    # Process each model
    for model_name in models:
        print(f"\n{'-'*50}")
        print(f"Processing model: {model_name}")
        print(f"{'-'*50}")
        
        # Initialize the model
        manager.initialize_model(model_name)
        
        # Train if requested
        if args.train:
            print(f"\nTraining {model_name} for {args.epochs} epochs...")
            manager.train_model(
                model_name, 
                epochs=args.epochs,
                save_weights=not args.no_save
            )
            print(f"Training completed for {model_name}")
        
        # Evaluate if requested
        if args.evaluate:
            print(f"\nEvaluating {model_name}...")
            accuracy, loss, report = manager.evaluate_model(model_name)
            print(f"Evaluation results for {model_name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Loss: {loss:.4f}")
            print("\nClassification Report:")
            print(report)
            
            # Plot confusion matrix for evaluated models
            model = manager.get_model(model_name)
            model.plot_confusion_matrix()
            model.plot_training_history()
    
    # Compare models if requested
    if args.compare and len(models) > 1:
        print("\nComparing models...")
        manager.compare_models(models)

if __name__ == "__main__":
    main()
    
    print("\nUsage examples:")
    print("  Train a specific model:")
    print("    python train_models.py --model custom_cnn --train --epochs 10")
    print("  Evaluate a model:")
    print("    python train_models.py --model densenet121 --evaluate")
    print("  Train and evaluate all models:")
    print("    python train_models.py --model all --train --evaluate")
    print("  Compare all models:")
    print("    python train_models.py --model all --train --evaluate --compare")
    print("  Visualize data distribution:")
    print("    python train_models.py --visualize") 