import tensorflow as tf 
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import numpy
from random import *

# Importeren van MNIST dataset
(input_train, target_train), (input_test, target_test) = emnist.load_data(type='letters')

#	Data normaliseren
input_train = tf.keras.utils.normalize(input_train, axis = 1 )

# Model opbouwen
model = tf.keras.models.Sequential()

# Flatten beelden => Array
model.add(tf.keras.layers.Flatten())

# Toevoegen van de eerste laag : relu, 128 neuronen
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Toevoegen van de tweede laag : relu, 128 neuronen
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Toevoegen van de derde laag  : softmax , 26 neuronen output 
model.add(tf.keras.layers.Dense(32, activation=tf.nn.softmax))

# Model compileren
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Trainen vd model in dree herhalingen tot wij een prestatie van 97% bekomen 

model.fit(input_train, target_train, epochs = 45 )	

# Model testen
val_loss, val_acc = model.evaluate(input_train, target_train)
print(val_loss, val_acc)
# werkelijk schatting  
for i in range(20): #   toen 20 sample	
	# random nummer generen om een foto uit de test dataset te kiezen, en testen
	random = randint(1,len(input_test))
	prediction = model.predict( [input_test] )[random]
	prediction = numpy.argmax([prediction], axis=1)
	plt.imshow(input_test[random], cmap = plt.cm.binary)
	plt.show()
	print (prediction)

"""
letter Number	 		Letter
1				 			A
2					 		B
3					 		C
4					 		D
5					 		E
6					 		F
7					 		G
8					 		H
9					 		I
10					 		J
11					 		K
12					 		L
13					 		M
14					 		N
15					 		O
16					 		P
17					 		Q
18					 		R
19					 		S
20					 		T
21					 		U
22					 		V
23					 		W
24					 		X
25					 		Y
26					 		Z
"""
print("EOF")