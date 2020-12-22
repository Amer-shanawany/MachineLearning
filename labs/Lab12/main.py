#!/usr/bin/python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy
# Data inlezen, als een python object 
cifar10 = keras.datasets.cifar10
# Data laden
(train_img, train_labels),(test_img, test_labels) = cifar10.load_data()

# Data normaliseren
train_img, test_img = train_img/255.0 , test_img/255.0

# lijst van namen
ctg_names = ['vliegtyug', 'auto', 'vogel', 'kat', 'hert', 'hond','kikker', 'paard', 'schip', 'vrachtwagen']

# 9 beelden toenen
def toon():
	plt.figure(figsize=(4,4))
	for i in range(9):
		plt.subplot(3, 3, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_img[i], cmap=plt.cm.binary)
		plt.xlabel(ctg_names[train_labels[i][0]])
	plt.show()

#toon()

# model opbouwen

model = keras.models.Sequential()
filtersL = 32 
kernelsL = (3,3)

# Eerste laag
model.add(layers.Conv2D(filtersL, kernelsL, activation='relu', input_shape=(32, 32, 3)))
# pixels polen/groeperen
poolsizeL=(2,2)

# Tweede laag
model.add(layers.MaxPool2D(poolsizeL))
# Derde laag
model.add(layers.Conv2D(filtersL, kernelsL, activation='relu'))

# model flattenen
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# model summary
print(model.summary())

# loss defineren
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']
# model compileren
model.compile(optimizer = optimizer, loss=loss, metrics=metrics)
epochs = 5
batchsize = 64

# model trainen
model.fit(train_img, train_labels, batch_size=batchsize)
# model evalueren
model.evaluate(test_img, test_labels, batch_size=batchsize)

# model testen
plt.imshow(test_img[1], cmap=plt.cm.binary)
plt.xlabel(ctg_names[test_labels[0][0]])
plt.show()

prediction = model.predict(test_img)
predict_img = (numpy.argmax(prediction[0]))
print(ctg_names[predict_img])

model.save('model_1')

print('EOF')