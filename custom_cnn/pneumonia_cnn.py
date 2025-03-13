import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2

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

# Data directories
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# Function to count images in each class
def count_images(directory):
    normal_dir = os.path.join(directory, 'NORMAL')
    pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
    
    normal_count = len([f for f in os.listdir(normal_dir) if not f.startswith('.')])
    pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')])
    
    return normal_count, pneumonia_count

# Count images
train_normal, train_pneumonia = count_images(train_dir)
val_normal, val_pneumonia = count_images(val_dir)
test_normal, test_pneumonia = count_images(test_dir)

print(f"Training set: {train_normal} normal, {train_pneumonia} pneumonia images")
print(f"Validation set: {val_normal} normal, {val_pneumonia} pneumonia images")
print(f"Test set: {test_normal} normal, {test_pneumonia} pneumonia images")

# Display dataset distribution
def plot_dataset_distribution():
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Train', 'Validation', 'Test'], 
                y=[train_normal + train_pneumonia, val_normal + val_pneumonia, test_normal + test_pneumonia])
    plt.title('Dataset Distribution')
    plt.ylabel('Number of Images')
    plt.show()

    # Class balance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Train Normal', 'Train Pneumonia', 'Val Normal', 'Val Pneumonia', 'Test Normal', 'Test Pneumonia'],
                y=[train_normal, train_pneumonia, val_normal, val_pneumonia, test_normal, test_pneumonia])
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to display sample images from each class
def display_sample_images(class_name, num_samples=4):
    class_dir = os.path.join(train_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if not f.startswith('.')][:num_samples]
    
    plt.figure(figsize=(15, 4))
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"{class_name}: {img.shape}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Set image size - smaller for CPU training
img_width, img_height = 156, 156  # Reduced size for CPU training
batch_size = 32

# Data augmentation for training (to help with generalization)
def setup_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test data only need rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Don't shuffle test data to preserve order for evaluation
    )
    
    return train_generator, validation_generator, test_generator

# Custom CNN architecture
def build_custom_cnn():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (156,156,3)))
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
    
    return model

# Train the model
def train_model(model, train_generator, validation_generator, epochs=20):
    # Calculate steps per epoch and validation steps
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    # Define callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('pneumonia_cnn_model.keras', save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
    ]
    
    # Train the model with fewer epochs for CPU training
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Evaluate model on test set
def evaluate_model(model, test_generator):
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    test_steps = np.ceil(test_generator.samples / batch_size).astype(int)
    predictions = model.predict(test_generator, steps=test_steps)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # True labels
    true_classes = test_generator.classes
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, 
                                target_names=['NORMAL', 'PNEUMONIA']))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'], 
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Visualize model predictions
def display_predictions(model, test_generator, num_images=8):
    # Reset generator to start from the beginning
    test_generator.reset()
    
    # Get one batch of test images
    batch_images, batch_labels = next(test_generator)
    
    # Make predictions
    predictions = model.predict(batch_images)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    # Display images with predictions
    plt.figure(figsize=(15, 12))
    for i in range(min(num_images, batch_images.shape[0])):
        plt.subplot(num_images // 4 + 1, 4, i+1)
        plt.imshow(batch_images[i])
        
        true_label = 'PNEUMONIA' if batch_labels[i] == 1 else 'NORMAL'
        pred_label = 'PNEUMONIA' if pred_classes[i] == 1 else 'NORMAL'
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label} ({predictions[i][0]:.2f})", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function to run the entire pipeline
def main():
    # Step 1: Display dataset distribution
    plot_dataset_distribution()
    
    # Step 2: Display sample images
    display_sample_images('NORMAL')
    display_sample_images('PNEUMONIA')
    
    # Step 3: Set up data generators
    train_generator, validation_generator, test_generator = setup_data_generators()
    
    # Step 4: Build the model
    model = build_custom_cnn()
    model.summary()
    
    # Step 5: Train the model
    history = train_model(model, train_generator, validation_generator)
    
    # Step 6: Plot training history
    plot_training_history(history)
    
    # Step 7: Evaluate model
    evaluate_model(model, test_generator)
    
    # Step 8: Display predictions
    display_predictions(model, test_generator)
    
    # Step 9: Save the model
    model.save('pneumonia_detection_model.h5')
    print("Model saved successfully!")

# If running as script (not imported)
if __name__ == "__main__":
    main() 