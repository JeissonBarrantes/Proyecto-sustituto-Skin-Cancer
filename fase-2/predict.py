import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
import os
import argparse
from loguru import logger


# Argument parser to handle command-line inputs for file paths
parser = argparse.ArgumentParser()
parser.add_argument('--img_input', required=True, type=str, help='Path to the folder containing images for prediction')
parser.add_argument('--predictions_file', required=True, type=str, help='A CSV file where predictions will be saved')
parser.add_argument('--model_file', required=True, type=str, help='Path to the stored model file (see train.py)')

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
model_file       = args.model_file
img_input        = args.img_input
predictions_file = args.predictions_file

model_path = model_file

# Check if the model file exists, load it if present
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
else:
    raise FileNotFoundError(f"The model file was not found at '{model_path}'.")

def preprocess_image(img_path, target_size=(256, 256)):
    """
    Loads and preprocesses an image from the given path.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for the image.

    Returns:
        np.array: Preprocessed image array ready for prediction.
    """
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Folder containing images for prediction
image_folder = img_input

# Verify if the folder exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"The folder '{image_folder}' was not found.")

# List all image files in the folder (supports .png, .jpg, .jpeg)
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not image_paths:
    raise FileNotFoundError(f"No images found in the folder '{image_folder}'.")

# List to store predictions
predictions = []

# Make predictions for each image
for img_path in image_paths:
    img_array = preprocess_image(img_path)
    
    # Make prediction using the loaded model
    prediction = model.predict(img_array)
    
    # Get the predicted class (index of the highest probability)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Store the image path and predicted class in the list
    predictions.append({"image_path": img_path, "predicted_class": predicted_class})
    
    # Print the prediction for each image
    print(f"Image: {img_path}, Predicted Class: {predicted_class}, Raw Prediction: {prediction}")

# Convert the list of predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to a CSV file
predictions_df.to_csv(predictions_file, index=False)

logger.info(f"Predictions saved in '{predictions_file}'.")
