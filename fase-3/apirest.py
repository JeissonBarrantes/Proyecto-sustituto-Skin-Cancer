from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from loguru import logger

# Inicializa la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apirest:app", host="0.0.0.0", port=8000, reload=True)

app = FastAPI()

model_path = "./model/skin_cancer_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "API is running"}

def preprocess_image(img_file, target_size=(256, 256)):
    """
    Carga y preprocesa una imagen para la predicción.

    Args:
        img_file: archivo de imagen cargado.
        target_size (tuple): tamaño objetivo para la imagen.

    Returns:
        np.array: Imagen preprocesada para predicción.
    """
    img = Image.open(img_file).convert("L").resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Añadir dimensiones de batch y canal
    img_array = img_array.astype("float32") / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verificar si el modelo está cargado
    if model is None:
        return {"error": "Model is not loaded"}

    try:
        # Leer y preprocesar la imagen
        img_array = preprocess_image(file.file)
        
        # Realizar la predicción
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        
        # Log de predicción
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Failed to process image and make prediction"}
