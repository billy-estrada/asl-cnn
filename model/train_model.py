import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
DATASET_PATH = '../dataset'
MODEL_PATH = 'models/asl_model.h5'


def sobel_preprocessing(img):
    img = img.astype(np.uint8)  

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.convertScaleAbs(magnitude)

    magnitude = magnitude.astype("float32") / 255.0  # Scale back to 0-1

    return np.expand_dims(magnitude, axis=-1)  # Adds the channel dimension

# 1. Data Loading with ImageDataGenerator
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    preprocessing_function=sobel_preprocessing,
)


train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

print(train_generator.class_indices)  # Print class indices for reference

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

import matplotlib.pyplot as plt

# Get one batch from the training generator
images, labels = next(train_generator)

# Plot the first 8 images
plt.figure(figsize=(12, 6))
for i in range(30):
    plt.subplot(5, 6, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')  # Remove the (1) channel dimension for display
    plt.axis('off')
plt.suptitle('Sobel + Augmented Images from Generator')
plt.show()

# 2. Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# 3. Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Training with Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 5. Plot Training Progress
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# 6. Save Final Model (optional if you didn’t use checkpoint)
model.save(MODEL_PATH)

print(f"\nModel trained and saved to: {MODEL_PATH}")

