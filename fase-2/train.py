import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse
from loguru import logger


# Argument parser for handling command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', required=True, type=str, help='A CSV file containing metadata for training')
parser.add_argument('--img_file', required=True, help='Path to the directory with images for training')
parser.add_argument('--model_file', required=True, type=str, help='Path to an existing stored model file (pkl)')
parser.add_argument('--overwrite_model', default=False, action='store_true', 
                    help='If set, overwrites the model file if it exists')

# Parse arguments
args = parser.parse_args()

metadata_file = args.metadata_file
model_file = args.model_file
img_file  = args.img_file
overwrite = args.overwrite_model

# Check if the model file exists and whether to overwrite it
if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"Overwriting existing model file {model_file}")
    else:
        logger.info(f"Model file {model_file} exists. Exiting. Use --overwrite_model option to overwrite.")
        exit(-1)
        

# Path to the data
train_dir = metadata_file
train_images_dir = img_file  # Directory with new images for training

# Training parameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = './skin_cancer_model.h5'

logger.info(f"Loading metadata")
# Load metadata from CSV
df = pd.read_csv(train_dir, usecols=['isic_id', 'target'])

logger.info(f"Getting images")
# Append full image path based on the metadata
df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(train_images_dir, f"{x}.jpg"))

logger.info(f"Preprocessing images")
# Filter out images that do not exist in the specified directory
df = df[df['image_path'].apply(os.path.exists)]

# Function to preprocess an image
def preprocess_image(img_path, target_size=(256, 256)):
    """
    Loads and preprocesses an image from the given path.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for the image.

    Returns:
        np.array: Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to load data (images and labels)
def load_data(df):
    """
    Loads and preprocesses images and their labels from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.

    Returns:
        tuple: A tuple containing the image arrays and the corresponding labels.
    """
    images = []
    labels = []
    for _, row in df.iterrows():
        img = preprocess_image(row['image_path'], target_size=IMG_SIZE)
        images.append(img)
        labels.append(int(row['target']))
    return np.vstack(images), np.array(labels)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)

# Load training and validation data
x_train, y_train = load_data(train_df)
x_val, y_val = load_data(val_df)

print(train_df)

# Load the existing model for retraining
model = load_model(MODEL_PATH)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert labels to one-hot encoding format
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Train the model
model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val)
)

logger.info(f"Model saved in {MODEL_PATH}")
# Save the updated model
model.save(MODEL_PATH)

logger.info("Done")
