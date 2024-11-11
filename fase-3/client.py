import requests

url = "http://localhost:8000/predict"
image_path = "./data/input_img/ISIC_0052367.jpg"  # Imagen de prueba

with open(image_path, "rb") as image_file:
    response = requests.post(url, files={"file": image_file})
    
print(response.json())
