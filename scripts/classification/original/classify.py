import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten  
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import pathlib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set the path to your local folder containing the agricultural crop folders
data_dir = "./agri_crops_clean"  
data_dir = pathlib.Path(data_dir)

# Image parameters
img_height, img_width = 224, 224  # ResNet50 default is 224x224
batch_size = 32
num_classes = 7  

# Load training data with augmentation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Load validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Get class names
class_names = train_ds.class_names
print(f"Classes: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Display some sample images from the dataset
plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('sample_crops.png')
plt.close()

# ResNet50 Model setup
print("Creating ResNet50 model...")
resnet_model = Sequential()

# Load pre-trained ResNet50 model
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                input_shape=(img_height, img_width, 3),
                                                pooling='avg',
                                                weights='imagenet')

# Freeze the pretrained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Add to our model
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())  
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(len(class_names), activation='softmax'))  # Dynamic number of classes

# Model summary
resnet_model.summary()

# Create a callback to save the best model
checkpoint = ModelCheckpoint(
    'best_crop_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# Compile model
resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Using sparse since we're using integer labels
    metrics=['accuracy']
)

# Train the model
epochs = 15
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Save the final model
resnet_model.save('final_crop_model.h5')

# Evaluate the model - Accuracy and Loss Curves
def plot_accuracy_loss(history):
    # Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim(0.4, 1)
    plt.grid(True)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'])
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()

# Plot training history
plot_accuracy_loss(history)

# Generate detailed model evaluation
print("Evaluating model on validation data...")
evaluation = resnet_model.evaluate(val_ds)
print(f"Validation Loss: {evaluation[0]:.4f}")
print(f"Validation Accuracy: {evaluation[1]:.4f}")

# Generate predictions for the validation set
all_labels = []
all_predictions = []

for images, labels in val_ds:
    predictions = resnet_model.predict(images)
    pred_classes = np.argmax(predictions, axis=1)
    
    all_labels.extend(labels.numpy())
    all_predictions.extend(pred_classes)

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Print classification report
cr = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
print("\nClassification Report:")
print(cr_df)
cr_df.to_csv('classification_report.csv')

# Function to make prediction on a single image
def predict_crop(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    
    predictions = resnet_model.predict(img_array)
    score = predictions[0]
    
    print(f"Image: {image_path}")
    print(f"Prediction probabilities: {score}")
    print(f"Predicted class: {class_names[np.argmax(score)]}")
    print(f"Confidence: {100 * np.max(score):.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[np.argmax(score)]}")
    plt.axis('off')
    plt.show()
    
    return class_names[np.argmax(score)], 100 * np.max(score)

# Sample prediction function - you can call this with a sample image path
# Example: predict_crop("/path/to/test_crop_image.jpg")

# If you want to test on some sample validation images
def test_sample_predictions():
    # Get a few sample images from validation set
    sample_count = 0
    plt.figure(figsize=(15, 10))
    
    for images, labels in val_ds:
        for i in range(min(6, len(images))):
            if sample_count >= 6:
                break
                
            # Save image temporarily
            img = images[i].numpy().astype("uint8")
            tmp_path = f"temp_img_{sample_count}.jpg"
            tf.keras.preprocessing.image.save_img(tmp_path, img)
            
            # Make prediction
            pred_class, confidence = predict_crop(tmp_path)
            true_class = class_names[labels[i].numpy()]
            
            # Display image with results
            ax = plt.subplot(2, 3, sample_count + 1)
            plt.imshow(img)
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
            plt.title(title)
            plt.axis("off")
            
            # Remove temporary file
            os.remove(tmp_path)
            sample_count += 1
        
        if sample_count >= 6:
            break
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

# Run sample prediction test
test_sample_predictions()
print("Model training and evaluation completed!")