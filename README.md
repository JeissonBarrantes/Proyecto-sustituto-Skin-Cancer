# Skin-Cancer

## Integrantes:

- **Natalia Andrea García Ríos**
  - *Cédula:* 1000655184

- **Jeisson Alexis Barrantes Toro**
  - *Cédula:* 1020484148

## FASE - 1
### PASOS PARA EJECUTAR

1. Las primeras celdas son de configuración e importación de librerías, en la celda con carga de archivo de google debes subir el archivo de Kaggle.json que obtienes como Key en la página oficial de Kaggle. Luego continua ejecutando las celdas hasta el apartado de Preprocessing.

2. En el apartado de Preprocessing puedes ejecutar tal cual está, esto ejecutará el preprocesamiento haciendo un ajuste de tamaño en las imágenes y además de ponerle un filtro para normalizarlas, puedes ajustar el tamaño de imágenes con cancer negativo que se usarán para entrenar donde se encuentra el uso de la función img_loader, especificamente modificar la linea de codigo tumor_b = img_loader(neg, train_images, neg.shape[0]//*250* ) cambiando el *250* por el valor que desees. No se recomienda debido a que el numero de imagenes dispuestas por la competencia de cancer positivo son muy pocas y podrías sesgar el modelo.

3. Por ultimo ejecutas cada una de las celdas del training y realizas la predicción. En está predicción encontrarás los resultados de forma preprocesada (para mayor facilidad) acerca de las predicciones que logro hacer el modelo respecto a las imagenes de los tipos de tumor. Siendo 0 benigno (negativo) y 1 maligno (positivo).

## FASE - 2
### PASOS PARA EJECUTAR

1. Debes crear la imagen de docker, para eso debes estar desde una terminal en la carpeta raiz del proyecto (directorio fase - 2), además recuerda que debes tener el modelo en predisposición para que sea leido por los scripts en este caso se espera tener un directorio model donde ira el modelo y ejecutar el comando *docker build -t skin_cancer_img .* y esperar a que se cree la imagen de docker. Recuerda que todos los pasos que seguira docker para crear la imagen están en el dockerfile. Esta imagen se creo apartir de una instanciada con tensorflow, permitiendo el uso de la mayoría de sus librerías necesarias para el proyecto.
   
2. Una vez creada la imagen de docker, debes ejecutar el comando *docker run -it skin_cancer_img* y entrarás en el entorno de ejecución de docker. Una vez aqui ya puedes ejecutar los scripts predict.py para realizar alguna predicción y train.py para realizar un entrenamiento.
   
4. Para ejecutar el predict.py y realizar una predicción debes seguir la proxima linea de comando:  
``  python predict.py  --img_input "./data/input_img"  --model_file "./model/skin_cancer_model.h5" --predictions_file "./predictions_file.csv" ``

5. (Opcional) Para ejecutar el train.py en caso de que quieras reentrenar el modelo o hacerle overtraning deberas ejecutar la siguiente linea de comando:
   ``python train.py --metadata_file "./data/train-metadata.csv" --img_file "./data/training" --model_file "./model/skin_cancer_model.h5" --overwrite_model (opcional)``

   
Recuerda que cada argumento esta direccionado a un path, podrás modificarlo pero debes tener en cuenta que la imagen lo tiene pensado de esa forma.

## FASE - 3
### PASOS PARA EJECUTAR


1. Debes crear la imagen de docker, para eso debes estar desde una terminal en la carpeta raiz del proyecto (directorio fase - 3), recuerda tener el modelo en carpeta model o donde prefieras teniendo en cuenta que debes ajustar el path y ejecutar el comando *docker build -t skin_cancer_img .* y esperar a que se cree la imagen de docker. Esta API fue creada con FastAPI y en el ``Dockerfile`` se encuentra el proceso de instalación de los requeriments y la exposicion del puerto ``8000``: CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0", "--port", "8000"]

2.Luego de crear el API en Docker, debes ejecutar ``python apirest.py`` para ejecutar el API y instanciar todo lo necesario para usar el endpoint de predict (Dentro de esta API se encuentra el preprocesado de que se le hará a todas las imágenes para ser usadas con el modelo) 
   
3. Para utilizar el endpoint de predict debes ejecutar en otra terminal ``python client.py`` dentro de este script esta el path de la imagen que puedes modificar si quieres probar otras imágenes.
