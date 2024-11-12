import requests
from loguru import logger
import pandas as pd

# Endpoint URL for prediction
predict_url = "http://localhost:8000/predict"
# Local path to the test image
image_path = "./data/input_img/ISIC_0052367.jpg"

# Endpoint URL for training
train_url = "http://localhost:8000/train"
# Paths to metadata CSV and ZIP file with training images
metadata_path = "./data/train-metadata.csv"
images_zip_path = "./data/training/images.zip"

# Flag to control model overwriting
overwrite_model = True

# Prediction request
try:
    with open(image_path, "rb") as image_file:
        logger.info("Sending prediction request...")
        predict_response = requests.post(predict_url, files={"file": image_file})
        # Log and print the prediction response
        logger.info("Prediction response received")
        print("Prediction Result:")
        print(predict_response.json())
except Exception as e:
    logger.error(f"Prediction request failed: {e}")

# Training request
try:
    with open(metadata_path, "rb") as metadata_file, open(images_zip_path, "rb") as images_zip:
        logger.info("Sending training request...")
        train_response = requests.post(
            train_url,
            files={
                "metadata_file": metadata_file,
                "img_file": images_zip
            },
            data={"overwrite_model": str(overwrite_model)}
        )
        # Log and print the training response
        logger.info("Training response received")
        print("Training Result:")
        print(train_response.json())
except Exception as e:
    logger.error(f"Training request failed: {e}")
