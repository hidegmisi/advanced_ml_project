import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.layers import Layer, Conv2D, Input, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Add parent directory to path to import BaseModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseModel

# Define custom layers for CapsNet
class PrimaryCapsule(Layer):
    def __init__(self, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)

    def call(self, inputs):
        s_norm = tf.norm(inputs, axis=-1, keepdims=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (inputs / (s_norm + 1e-7))

class DigitCapsule(Layer):
    def __init__(self, **kwargs):
        super(DigitCapsule, self).__init__(**kwargs)

    def call(self, inputs):
        s = tf.reduce_sum(inputs, axis=1)
        return PrimaryCapsule()(s)

class Mapping(Layer):
    def __init__(self, num_pc_outputs, num_classes, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.num_pc_outputs = num_pc_outputs
        self.num_classes = num_classes
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.num_pc_outputs, self.num_classes, 8, 16),
            initializer='random_normal',
            trainable=True,
            name='mapping_weights'
        )
        super(Mapping, self).build(input_shape)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.W)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes, 16)

class Reshape3D(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Reshape3D, self).__init__(**kwargs)
        self.target_shape = target_shape
        
    def call(self, inputs):
        return tf.reshape(inputs, [-1, *self.target_shape])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.target_shape)

class CapsulePooling(Layer):
    """Layer to perform global average pooling on capsules"""
    def __init__(self, **kwargs):
        super(CapsulePooling, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

class CapsNet(BaseModel):
    def __init__(self, img_size=(128, 128), batch_size=16):
        super().__init__(img_size=img_size, batch_size=batch_size)
        self.model_name = "capsnet"
        self.NUM_CHANNELS = 5
        self.NUM_PC_OUTPUTS = self.NUM_CHANNELS * 36
        self.NUM_CLASSES = 2
        
    def setup_data_generators(self, train_dir="data/train", val_dir=None, test_dir="data/test"):
        """Set up data generators for training, validation, and testing.
        Override parent method to use categorical mode for labels.
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory (optional)
            test_dir (str): Path to test data directory
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2 if val_dir is None else 0
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators with categorical class mode
        self.train_generator = train_datagen.flow_from_directory(
            train_dir, 
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            subset='training' if val_dir is None else None
        )
        
        if val_dir is None:
            self.val_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.IMG_SIZE,
                batch_size=self.BATCH_SIZE,
                class_mode='categorical',
                subset='validation'
            )
        else:
            self.val_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=self.IMG_SIZE,
                batch_size=self.BATCH_SIZE,
                class_mode='categorical'
            )
            
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Found {self.test_generator.samples} test images")
    
    def build_model(self):
        """Build the CapsNet architecture"""
        from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
        
        # Define the input
        inputs = Input(shape=(*self.IMG_SIZE, 3), name="input_layer")
        
        # Convolutional layers with batch normalization
        x = Conv2D(256, kernel_size=9, strides=1, padding='valid', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, kernel_size=5, strides=2, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Final convolutional layer
        x = Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Flatten and add dense layers for classification
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Display model summary
        self.model.summary()
        
        return self.model
    
    def evaluate(self):
        """Evaluate the model - override to handle categorical labels"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Evaluate on test data
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get true labels and predictions
        true_labels = self.test_generator.classes
        predictions = self.model.predict(self.test_generator)
        predicted_labels = np.argmax(predictions, axis=1)

        # Generate classification report
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=list(self.test_generator.class_indices.keys())
        )
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predicted_labels)
        
        return test_acc, test_loss, report
    
    def display_predictions(self, num_images=8):
        """Visualize model predictions - override to handle categorical labels"""
        import matplotlib.pyplot as plt
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Reset generator to start from the beginning
        self.test_generator.reset()
        
        # Get one batch of test images
        batch_images, batch_labels = next(self.test_generator)
        
        # Make predictions
        predictions = self.model.predict(batch_images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(batch_labels, axis=1)
        
        # Display images with predictions
        plt.figure(figsize=(15, 12))
        class_names = list(self.test_generator.class_indices.keys())
        
        for i in range(min(num_images, batch_images.shape[0])):
            plt.subplot(num_images // 4 + 1, 4, i+1)
            plt.imshow(batch_images[i])
            
            true_label = class_names[true_classes[i]]
            pred_label = class_names[pred_classes[i]]
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f"True: {true_label}\nPred: {pred_label} ({predictions[i][pred_classes[i]]:.2f})", color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

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