import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

# Check TensorFlow version and device
print("TensorFlow version:", tf.__version__)
print("Using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)

# Configure GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class CustomCNNModel(BaseModel):
    def __init__(self, img_size=(156, 156), batch_size=32):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "custom_cnn"
        
    def count_images(self, directory):
        normal_dir = os.path.join(directory, 'NORMAL')
        pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
        
        normal_count = len([f for f in os.listdir(normal_dir) if not f.startswith('.')])
        pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')])
        
        return normal_count, pneumonia_count
    
    def setup_data_generators(self, train_dir="train", val_dir="val", test_dir="test"):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Count images in each set
        if os.path.exists(train_dir):
            train_normal, train_pneumonia = self.count_images(train_dir)
            print(f"Training set: {train_normal} normal, {train_pneumonia} pneumonia images")
        
        if os.path.exists(val_dir):
            val_normal, val_pneumonia = self.count_images(val_dir)
            print(f"Validation set: {val_normal} normal, {val_pneumonia} pneumonia images")
        
        if os.path.exists(test_dir):
            test_normal, test_pneumonia = self.count_images(test_dir)
            print(f"Test set: {test_normal} normal, {test_pneumonia} pneumonia images")
        
        # Call the parent method to set up the data generators
        super().setup_data_generators(train_dir, val_dir, test_dir)
    
    def build_model(self):
        model = models.Sequential()
        
        model.add(layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (*self.IMG_SIZE, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(units = 128 , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units = 1 , activation = 'sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.model.summary()
        
        return model
    
    def display_sample_images(self, class_name, num_samples=4):
        class_dir = os.path.join(self.train_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if not f.startswith('.')][:num_samples]
        
        plt.figure(figsize=(15, 4))
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"{class_name}: {img.shape}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    model = CustomCNNModel()
    model.display_dataset_distribution()
    model.display_sample_images('NORMAL')
    model.display_sample_images('PNEUMONIA')
    model.setup_data_generators()
    model.build_model()
    model.train(epochs=20)
    model.plot_training_history()
    model.evaluate()
    model.display_predictions()
    model.save_model()
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 