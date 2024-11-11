# Usar una imagen base de Python con TensorFlow preinstalado
FROM tensorflow/tensorflow:2.17.0 

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Exponer puerto (si tienes alguna API o interfaz que desees incluir)
EXPOSE 8000

# Comando para correr FastAPI con Uvicorn
CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0", "--port", "8000"]




