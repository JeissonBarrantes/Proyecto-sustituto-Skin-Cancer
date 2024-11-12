from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import tempfile

"""
FastAPI script for a skin cancer classification REST API.

This script creates a FastAPI application with endpoints for training and predicting skin cancer classifications 
based on image inputs. It utilizes a TensorFlow-based Convolutional Neural Network (CNN) model to classify 
images as benign ("benigno") or malignant ("maligno").

Modules and Libraries:
- FastAPI: Framework to create the REST API.
- PIL (Pillow): For image processing.
- TensorFlow & Keras: To build and load the CNN model.
- NumPy: To handle image arrays and numerical data.
- Pandas: For handling metadata in CSV format.
- Loguru: For logging messages and errors.
- Sklearn: For data splitting.
- Shutil & Tempfile: For managing file storage and unpacking images.

Key Constants:
- IMG_SIZE: Target size for input images, set to 256x256 pixels.
- BATCH_SIZE: Batch size for training, set to 32.
- EPOCHS: Number of training epochs, set to 10.
- MODEL_PATH: File path for saving/loading the trained model.
- DATA_DIR: Directory for storing training data.

Endpoints:
1. GET "/":
   - Simple health check to confirm the API is running.

2. POST "/train":
   - Trains or retrains the model using metadata and image files uploaded by the user.
   - Parameters:
       - metadata_file: CSV file with metadata containing image IDs and labels.
       - img_file: ZIP file with images matching the IDs in the metadata.
       - overwrite_model: Flag to control whether to overwrite the existing model file.
   - Returns:
       - JSON response indicating the success of the training process.

3. POST "/predict":
   - Performs prediction on a single uploaded image file.
   - Parameters:
       - file: Image file (single) to be classified.
   - Returns:
       - JSON response containing the predicted class ("benigno" or "maligno") and the model's confidence.

Functions:
- preprocess_image(img_file, target_size): Loads and preprocesses an image file for model input.
- load_data(df): Loads image data and labels from metadata for training and validation.
"""

# Initialize the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apirest:app", host="0.0.0.0", port=8000, reload=True)

app = FastAPI()

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "./model/skin_cancer_model.h5"
DATA_DIR = "./data/training"

# Load the model
try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    """Simple health check to confirm the API is running."""
    return {"message": "API is running"}

def preprocess_image(img_file, target_size=(256, 256)):
    """
    Load and preprocess an image for prediction.

    Args:
        img_file: The uploaded image file.
        target_size (tuple): Target size for the image.

    Returns:
        np.array: Preprocessed image ready for model input.
    """
    img = Image.open(img_file).convert("L").resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    img_array = img_array.astype("float32") / 255.0
    return img_array

def load_data(df):
    """
    Load image data and labels from a DataFrame for model training and validation.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.

    Returns:
        Tuple[np.array, np.array]: Arrays of images and labels.
    """
    images = []
    labels = []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"], target_size=IMG_SIZE)
        images.append(img)
        labels.append(int(row["target"]))
    return np.vstack(images), np.array(labels)

@app.post("/train")
async def train_model(metadata_file: UploadFile = File(...), img_file: UploadFile = File(...), overwrite_model: bool = False):
    """
    Train or retrain the model using provided metadata and images.

    Args:
        metadata_file (UploadFile): CSV file with metadata.
        img_file (UploadFile): ZIP file containing images.
        overwrite_model (bool): Whether to overwrite the existing model.

    Returns:
        dict: Response indicating training success or failure.
    """
    global model
    try:
        # Save metadata CSV
        metadata_path = os.path.join(DATA_DIR, "metadata.csv")
        with open(metadata_path, "wb") as f:
            f.write(await metadata_file.read())

        # Save the ZIP file temporarily and then unpack
        with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
            temp_zip.write(await img_file.read())
            temp_zip_path = temp_zip.name

        img_dir = os.path.join(DATA_DIR, "images")
        shutil.unpack_archive(temp_zip_path, img_dir, "zip")
        
        # Remove the temporary ZIP file
        os.remove(temp_zip_path)
        
        # Load and filter metadata
        df = pd.read_csv(metadata_path, usecols=["isic_id", "target"])
        df["image_path"] = df["isic_id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
        df = df[df["image_path"].apply(os.path.exists)]
        
        # Split into training and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
        x_train, y_train = load_data(train_df)
        x_val, y_val = load_data(val_df)
        
        # Check if the model should be overwritten
        if os.path.exists(MODEL_PATH) and not overwrite_model:
            return {"message": "Model file already exists. Use overwrite_model to overwrite."}
        
        # Define the model if it hasn't been loaded yet
        if model is None:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
        
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
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
        
        # Save the trained model
        model.save(MODEL_PATH)
        logger.info(f"Model trained and saved successfully at {MODEL_PATH}")
        
        return {"message": "Model trained and saved successfully"}
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": "Failed to train the model"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the class of a skin lesion in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file for prediction.

    Returns:
        dict: Predicted class ("benigno" or "maligno") and confidence score.
    """
    # Check if the model is loaded
    if model is None:
        return {"error": "Model is not loaded"}

    try:
        # Read and preprocess the image
        img_array = preprocess_image(file.file)
        
        # Perform the prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        
        # Convert predicted class to text
        class_label = "benigno" if predicted_class == 0 else "maligno"
        
        # Log the prediction
        logger.info(f"Predicted class: {class_label}, Confidence: {confidence}")
        
        return {"predicted_class": class_label, "confidence": confidence}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Failed to process image and make prediction"}
