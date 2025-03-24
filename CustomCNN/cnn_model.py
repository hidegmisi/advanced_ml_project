import os
import tensorflow as tf
from tensorflow.keras import layers, models
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

# Configure GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class CustomCNNModel(BaseModel):
    def __init__(self, img_size=(150, 150), batch_size=32):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "custom_cnn"
    
    def build_model(self):
        """Build a simple CNN model"""
        model = models.Sequential()
        
        # Simple CNN architecture
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(*self.IMG_SIZE, 3)))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(128, (3,3), activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.model.summary()
        
        return model

if __name__ == "__main__":
    # Create model instance
    model = CustomCNNModel()
    
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