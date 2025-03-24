import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# Add parent directory to path to import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

class CapsNetModel(BaseModel):
    def __init__(self, img_size=(128, 128), batch_size=16):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "capsnet"
        
    def build_model(self):
        """Build a simplified CapsNet-inspired model"""
        # Define the input
        inputs = Input(shape=(*self.IMG_SIZE, 3))
        
        # Convolutional layers
        x = Conv2D(64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=3, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Flatten and add dense layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

# Main function to run the entire pipeline
def main():
    # Create model instance
    model = CapsNet()
    
    # Display dataset distribution
    model.display_dataset_distribution()
    
    # Display sample images
    model.display_sample_images()
    
    # Set up data generators
    model.setup_data_generators()
    
    # Build the model
    model.build_model()
    
    # Train the model
    model.train(epochs=10)
    
    # Plot training history
    model.plot_training_history()
    
    # Evaluate model
    model.evaluate()
    
    # Display predictions
    model.display_predictions()
    
    # Save the model
    model.save_model()
    print("Model saved successfully!")

# If running as script (not imported)
if __name__ == "__main__":
    main() 