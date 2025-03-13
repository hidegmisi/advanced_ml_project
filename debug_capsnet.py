import tensorflow as tf
import os
from CapsNet.caps_model import CapsNet

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# Create directories if they don't exist
os.makedirs('weights', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Create CapsNet model
print("\nInitializing CapsNet model...")
capsnet = CapsNet(img_size=(28, 28), batch_size=16)

# Build model without data (just to test model architecture)
print("\nBuilding CapsNet model...")
try:
    capsnet.build_model()
    print("Model built successfully!")
except Exception as e:
    print(f"Error building model: {e}")

# Print model information
if capsnet.model is not None:
    print("\nModel summary:")
    capsnet.model.summary()
else:
    print("\nModel could not be built.") 