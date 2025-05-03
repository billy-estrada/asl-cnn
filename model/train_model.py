import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold
import cv2
import shutil

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
DATASET_PATH = '../dataset'
MODEL_PATH = 'models/asl_model.h5'

OUTPUT_DIR = '../dataset_split'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# logic to help split datasets by tagged identifier
data = []
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        for fname in os.listdir(label_path):
            if fname.endswith('.jpg'):
                parts = fname.split('_')  # Format: {label}_be_{tag}_{id}.jpg
                tag = parts[2]
                full_path = os.path.join(label_path, fname)
                data.append((label, tag, full_path))

df = pd.DataFrame(data, columns=['label', 'tag', 'path'])

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

def copy_split(df_split, base_dir):
    for _, row in df_split.iterrows():
        label_dir = os.path.join(base_dir, row['label'])
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(row['path'], os.path.join(label_dir, os.path.basename(row['path'])))

copy_split(train_df, TRAIN_DIR)
copy_split(val_df, VAL_DIR)


def sobel_preprocessing(img):
    img = img.astype(np.uint8)  
    
    denoised = cv2.medianBlur(img, ksize=3)
    
    sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.convertScaleAbs(magnitude)
    
    _, strong_edges = cv2.threshold(denoised, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    result = cv2.bitwise_and(denoised, denoised, mask=strong_edges)
    
    # result = cv2.bitwise_and(blurred, blurred, mask=thresh)

    magnitude = result.astype("float32") / 255.0  # Scale back to 0-1

    return np.expand_dims(magnitude, axis=-1)  # Adds the channel dimension


# Data loading with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range=0.1,
    preprocessing_function=sobel_preprocessing,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=sobel_preprocessing
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

print(train_generator.class_indices)  # Print class indices for reference

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

import matplotlib.pyplot as plt

# Get one batch from the training generator
images, labels = next(train_generator)

# see post process images
plt.figure(figsize=(12, 6))
for i in range(30):
    plt.subplot(5, 6, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')  # Remove the (1) channel dimension for display
    plt.axis('off')
plt.suptitle('Sobel + Augmented Images from Generator')
plt.show()

# 2. build the CNN model
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

# 3. compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. stop after plateau
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
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

