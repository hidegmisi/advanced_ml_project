# Pneumonia Detection Models

This repository contains implementations of three deep learning models for pneumonia detection from chest X-ray images:

1. **DenseNet121** - A pre-trained model fine-tuned for pneumonia detection
2. **Custom CNN** - A custom convolutional neural network architecture
3. **CapsNet** - A Capsule Network implementation for pneumonia classification

## Directory Structure

```
├── CapsNet/
│   ├── caps_model.py           # CapsNet model implementation
│   └── caps.ipynb              # Original notebook (empty)
├── custom_cnn/
│   ├── pneumonia_cnn.py        # Original CNN implementation
│   └── pneumonia_cnn_model.py  # New class-based CNN implementation
├── DenseNet121/
│   ├── densenet121_pneumonia.py # DenseNet121 model implementation
│   └── notebook.ipynb          # Original notebook
├── utils/
│   └── model_manager.py        # Utility for managing models
├── comparisons/                # Directory for saving comparison results
└── data/                       # Data directory (not included)
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Models Overview

### DenseNet121

A pre-trained DenseNet121 model fine-tuned for pneumonia detection. DenseNet models connect each layer to every other layer in a feed-forward fashion, which helps alleviate the vanishing gradient problem, strengthens feature propagation, and reduces the number of parameters.

### Custom CNN

A custom-designed convolutional neural network with multiple convolutional blocks, batch normalization, and dropout layers. The architecture is designed specifically for the pneumonia detection task.

### CapsNet (Capsule Network)

A Capsule Network implementation based on the paper "Dynamic Routing Between Capsules" by Sabour et al. Capsule Networks use groups of neurons (capsules) to represent spatial relationships between features, offering a more robust approach to handling variations in image features.

## Usage

### Using the ModelManager

The `ModelManager` class provides a unified interface to work with all three models:

```python
from utils.model_manager import ModelManager

# Initialize model manager
manager = ModelManager(data_dir="path/to/data")

# Initialize all models
manager.initialize_all_models()

# Or initialize a specific model
custom_cnn = manager.initialize_model("custom_cnn")

# Train a model
history = manager.train_model("densenet121", epochs=10)

# Evaluate a model
acc, loss, report = manager.evaluate_model("custom_cnn")

# Compare models
results = manager.compare_models()

# Compare training histories
manager.compare_training_history()
```

### Using Individual Models

Each model can also be used independently:

```python
# Using Custom CNN
from custom_cnn.pneumonia_cnn_model import CustomCNNModel

model = CustomCNNModel()
model.setup_data_generators(train_dir="data/train", val_dir="data/val", test_dir="data/test")
model.build_model()
history = model.train(epochs=10)
model.save_training_history("history/custom_cnn_history.json")
model.plot_training_history()

# Similar usage for other models...
```

## Data Preparation

The models expect data to be organized in the following structure:

```
data/
├── train/
│   ├── NORMAL/       # Normal chest X-ray images
│   └── PNEUMONIA/    # Pneumonia chest X-ray images
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- pandas
- scikit-learn
- seaborn

## Model Weight and History Saving

Each model implementation includes functionality to:

1. Save model weights during training (using callbacks)
2. Save and load training history (as JSON or pickle files)
3. Load pre-trained weights for inference

Example:
```python
# Save weights
model.save_weights("weights/custom_cnn_weights.h5")

# Load weights
model.load_weights("weights/custom_cnn_weights.h5")

# Save training history
model.save_training_history("history/custom_cnn_history.json")

# Load and visualize training history
model.plot_training_history("history/custom_cnn_history.json")
```

## Model Comparison

The `ModelManager` provides tools to compare performance across models:

1. Accuracy and loss comparison
2. Training history visualization
3. Export of comparison results as CSV and PNG

Results are saved in the `comparisons/` directory.
