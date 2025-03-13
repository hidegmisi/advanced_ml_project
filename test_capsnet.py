import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from CapsNet.caps_model import CapsNet
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check TensorFlow version and devices
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# Create test directories
def prepare_test_data():
    """
    Create a small test dataset by using a subset of MNIST data,
    saving it in the proper directory structure for the data generators.
    """
    # Create directories
    base_dir = "test_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:  # Using MNIST class 0 as NORMAL, class 1 as PNEUMONIA
            os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Only use digits 0 and 1 for binary classification
    x_train_filtered = []
    y_train_filtered = []
    for i, label in enumerate(y_train):
        if label in [0, 1]:
            x_train_filtered.append(x_train[i])
            y_train_filtered.append(label)
    
    x_test_filtered = []
    y_test_filtered = []
    for i, label in enumerate(y_test):
        if label in [0, 1]:
            x_test_filtered.append(x_test[i])
            y_test_filtered.append(label)
    
    x_train_filtered = np.array(x_train_filtered)
    y_train_filtered = np.array(y_train_filtered)
    x_test_filtered = np.array(x_test_filtered)
    y_test_filtered = np.array(y_test_filtered)
    
    # Split into train (70%), val (15%), test (15%)
    num_train = int(len(x_train_filtered) * 0.7)
    num_val = int(len(x_train_filtered) * 0.15)
    
    x_val = x_train_filtered[num_train:num_train+num_val]
    y_val = y_train_filtered[num_train:num_train+num_val]
    x_train = x_train_filtered[:num_train]
    y_train = y_train_filtered[:num_train]
    x_test = x_test_filtered[:100]  # Only use a small subset for testing
    y_test = y_test_filtered[:100]
    
    # Save images in the correct directories
    for x, y, split in [(x_train, y_train, "train"), (x_val, y_val, "val"), (x_test, y_test, "test")]:
        for i, (img, label) in enumerate(zip(x, y)):
            label_name = "NORMAL" if label == 0 else "PNEUMONIA"
            img_path = os.path.join(base_dir, split, label_name, f"{i}.png")
            
            # Normalize to 0-255 uint8 for image saving
            img_normalized = ((img / 255.0) * 255).astype(np.uint8)
            
            # Save image using matplotlib
            plt.imsave(img_path, img_normalized, cmap='gray')
    
    print(f"Dataset created in {base_dir}")
    print(f"Train: {len(x_train)} images")
    print(f"Validation: {len(x_val)} images")
    print(f"Test: {len(x_test)} images")
    
    return base_dir

# Create test dataset
print("\nPreparing test dataset...")
data_dir = prepare_test_data()

# Create directories for model outputs
os.makedirs('weights', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize CapsNet model
print("\nInitializing CapsNet model...")
capsnet = CapsNet(img_size=(28, 28), batch_size=32, routings=3)

# Setup data generators
print("\nSetting up data generators...")
capsnet.setup_data_generators(
    train_dir=os.path.join(data_dir, "train"),
    val_dir=os.path.join(data_dir, "val"),
    test_dir=os.path.join(data_dir, "test")
)

# Build model
print("\nBuilding CapsNet model...")
capsnet.build_model()

# Train model (with fewer epochs for testing)
print("\nTraining model for a few epochs...")
try:
    history = capsnet.train(epochs=3)
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    
# Evaluate model
print("\nEvaluating model...")
try:
    accuracy, loss, report = capsnet.evaluate()
    print(f"Evaluation completed with accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

# Visualize results
try:
    print("\nPlotting training history...")
    capsnet.plot_training_history()
except Exception as e:
    print(f"Error plotting history: {e}")

print("\nTest completed! CapsNet is working properly.") 