                                                                                                                                                                     import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#se guardan las rutas de las carpetas de entrenamiento y validacion
entrenamiento = '../data/dataset/entrenamiento'
validacion = '../data/dataset/validacion'

listaEntrenamiento = os.listdir(entrenamiento) # Obtiene la lista de nombres de archivos en la carpeta de entrenamiento
listaValidacion = os.listdir(validacion) # Obtiene la lista de nombres de archivos en la carpeta de validación

ancho, alto = 224, 224

etiquetas = [] # Lista para almacenar las etiquetas de las imágenes
fotos = [] # Lista para almacenar las imágenes
datos_entrenamiento = [] # Lista para almacenar los datos de entrenamiento
con = 0

etiquetas2 = [] # Lista para almacenar las etiquetas de las imágenes de validación
fotos2 = [] # Lista para almacenar las imágenes de validación
datos_validacion = [] # Lista para almacenar los datos de validación
con2 = 0
                                                                      
# Lectura de las imágenes de las carpetas de entrenamiento
for nameDir in listaEntrenamiento:
    nombre = entrenamiento + "/" + nameDir
    for nameFile in os.listdir(nombre):
        etiquetas.append(con) # Agrega la etiqueta (con) a la lista de etiquetas
        img = cv2.imread(nombre + "/" + nameFile, 0) # Lee la imagen en escala de grises
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC) # Cambia el tamaño de la imagen
        img = img.reshape(ancho, alto, 1) # Cambia la forma de la imagen
        datos_entrenamiento.append([img, con]) # Agrega la imagen y su etiqueta a los datos de entrenamiento
        fotos.append(img) # Agrega la imagen a la lista de fotos
    con += 1

# Lectura de las imágenes de las carpetas de validación
for nameDir2 in listaValidacion:
    nombre2 = validacion + "/" + nameDir2
    for nameFile2 in os.listdir(nombre2):
        etiquetas2.append(con2) # Agrega la etiqueta (con2) a la lista de etiquetas de validación
        img2 = cv2.imread(nombre2 + "/" + nameFile2, 0) # Lee la imagen en escala de grises
        img2 = cv2.resize(img2, (ancho, alto), interpolation=cv2.INTER_CUBIC) # Cambia el tamaño de la imagen
        img2 = img2.reshape(ancho, alto, 1) # Cambia la forma de la imagen
        datos_validacion.append([img2, con2]) # Agrega la imagen y su etiqueta a los datos de validación
        fotos2.append(img2) # Agrega la imagen a la lista de fotos de validación
    con2 += 1

# Normalización de las imágenes
fotos = np.array(fotos).astype('float') / 255
print(fotos.shape)
fotos2 = np.array(fotos2).astype('float') / 255
print(fotos2.shape)

etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

# Creación de un generador de imágenes para el aumento de datos
# se generaran la misma cantidad de imagenes que tiene el dataset de entrenamiento por las 7 caracteristicas que se le aplicaran
imgTrainGen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
)
imgTrainGen.fit(fotos)

# Visualización de imágenes generadas
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=1, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[0], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)



# Se define otro modelo de red neuronal convolucional llamado 'ModeloCNN2' utilizando Sequential de TensorFlow Keras.
# Este modelo es similar a 'ModeloCNN' pero con una capa de dropout entre las capas convolucionales y las capas densas para prevenir el sobreajuste.
ModeloCNN2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ancho, alto, 1)), # Capa convolucional con 32 filtros y activación ReLU
    tf.keras.layers.MaxPooling2D((2, 2)), # Capa de max-pooling para reducir la dimensionalidad
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Capa convolucional con 64 filtros y activación ReLU
    tf.keras.layers.MaxPooling2D((2, 2)), # Capa de max-pooling para reducir la dimensionalidad
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # Capa convolucional con 128 filtros y activación ReLU
    tf.keras.layers.MaxPooling2D((2, 2)), # Capa de max-pooling para reducir la dimensionalidad
    tf.keras.layers.Dropout(0.5), # Capa de dropout con tasa de dropout de 0.5 para prevenir el sobreajuste
    tf.keras.layers.Flatten(), # Capa de aplanamiento para preparar la salida de las capas convolucionales para las capas densas
    tf.keras.layers.Dense(256, activation='relu'), # Capa densa con 256 neuronas y activación ReLU
    tf.keras.layers.Dense(1, activation='sigmoid') # Capa de salida con 1 neurona y activación sigmoide
])

# Compilación de modelos
# Se compilan los modelos utilizando el optimizador 'adam', la función de pérdida 'binary_crossentropy' (apropiada para problemas de clasificación binaria)
# y se monitorea la métrica de precisión ('accuracy').

ModeloCNN2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento de modelos
# Se entrena cada modelo durante 10 épocas utilizando los datos generados por el generador de imágenes 'imgTrain'.
# Se utiliza un tamaño de lote de 32 imágenes y se valida el modelo utilizando los datos de validación '(fotos2, etiquetas2)'.
# Además, se utilizan callbacks para guardar los datos de la ejecución del modelo en TensorBoard.

ModeloCNN2.fit(imgTrain, batch_size=32, epochs=10, validation_data=(fotos2, etiquetas2), callbacks=[TensorBoard(log_dir='./logs/ModeloCNN2')])







