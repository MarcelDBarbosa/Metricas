#Importar Bibliotecas
from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

#Carregar dados do MNist, dividir o conjunto em treino e teste,reformatar e normalizar as imagens
(train_images, train_labels), (test_images,test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))
train_images, test_images = train_images/255.0, test_images/255.0

classes=[0,1,2,3,4,5,6,7,8,9]

#Definição de 5 camadas profundas e uma camada de saída com 10 neurônios
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

#Definição dos hiperparâmetros da rede neural
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Treinamento da rede
model.fit(x=train_images,y=train_labels,epochs=5,validation_data=(test_images,test_labels))

#Define variáveis para os valores reais e os previstos
y_true=test_labels
y_pred=model.predict(test_images)

#Transforma a predição em porcentagem num vetor com o valor mais provável do algarismo
y_predf=np.argmax(y_pred,axis=1) 

#Calcula a matriz de confusão
con_mat = tf.math.confusion_matrix(labels=y_true,predictions=y_predf).numpy()
print(con_mat)
