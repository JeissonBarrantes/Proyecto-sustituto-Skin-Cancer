# Usar una imagen base de Python con TensorFlow preinstalado
FROM tensorflow/tensorflow:2.17.0 

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias adicionales necesarias
RUN pip install --no-cache-dir pandas

# Instalar Pillow
RUN pip install pillow
RUN pip install loguru

# Exponer puerto (si tienes alguna API o interfaz que desees incluir)
EXPOSE 5000





