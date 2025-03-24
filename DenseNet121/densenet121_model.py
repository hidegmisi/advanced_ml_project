import tensorflow as tf
import os
import sys
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Add parent directory to path to import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

class DenseNet121Model(BaseModel):
    def __init__(self, img_size=(150, 150), batch_size=32):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "densenet121"
        
    def build_model(self):
        """Build the DenseNet121 model with custom top layers."""
        # Load pre-trained DenseNet121 without the top classification layer
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(*self.IMG_SIZE, 3))

        # Freeze the base model
        base_model.trainable = False

        # Add Custom Layers on top
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=output)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        self.model.summary()
        return self.model

if __name__ == "__main__":
    # Create model instance
    model = DenseNet121Model()
    
    # Show dataset distribution
    model.show_data_distribution()
    
    # Setup data generators and build model
    model.setup_data_generators()
    model.build_model()
    
    # Train and evaluate
    model.train(epochs=10)
    model.evaluate()
    
    # Visualize results
    model.plot_training_history()
    model.plot_confusion_matrix()
    
    # Save the model
    model.save_model()
    
    print("Model saved successfully!") 