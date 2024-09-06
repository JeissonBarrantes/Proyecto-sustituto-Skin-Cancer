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
