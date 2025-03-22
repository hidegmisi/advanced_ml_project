import tensorflow as tf
import os
import sys
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Add parent directory to path to import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

class DenseNet121Model(BaseModel):
    def __init__(self, img_size=(224, 224), batch_size=32):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "densenet121"
        
    def build_model(self):
        """Build the DenseNet121 model with custom top layers."""
        # Load pre-trained DenseNet121 without the top classification layer
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*self.IMG_SIZE, 3))

        # Freeze the base model (so its weights won't be updated during training)
        base_model.trainable = False

        # Add Custom Layers on top
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.1)(x)
        output = Dense(1, activation="sigmoid")(x)  # Binary classification

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=output)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Display model summary
        self.model.summary()
        return self.model

# Example usage
if __name__ == "__main__":
    # Create model
    densenet = DenseNet121Model()
    
    # Setup data generators
    densenet.setup_data_generators()
    
    # Build model
    densenet.build_model()
    
    # Train model
    densenet.train(epochs=10)
    
    # Evaluate model
    densenet.evaluate()
    
    # Plot training history
    densenet.plot_training_history()
    
    # Display predictions
    densenet.display_predictions()
    
    # Save model
    densenet.save_model()
    
    print("Model saved successfully!") 