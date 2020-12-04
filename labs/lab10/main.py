import tensorflow as tf 
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy

# Importeren van MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 	Image toenen 
#plt.imshow(x_train[0])
#plt.show()
# 	Image toenen in grayscale 
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#print(x_train[0])

#	Data normaliseren
x_train = tf.keras.utils.normalize(x_train, axis = 1 )

# Image toenen na de normalizastie 
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()

# Model opbouwen
model = tf.keras.models.Sequential()

# Flatten beelden => Array
model.add(tf.keras.layers.Flatten())

# Toevoegen van de eerste laag : relu, 128 neuronen
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Toevoegen van de tweede laag : relu, 128 neuronen
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Toevoegen van de derde laag  : softmax , 10 neuronen output
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Model compileren
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Trainen vd model in dree herhalingen 
model.fit(x_train, y_train, epochs = 3 )

# Model testen
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# werkelijk schatting 

prediction = model.predict( [x_test] )[1]
prediction = numpy.argmax([prediction], axis=1)
plt.imshow(x_test[1], cmap = plt.cm.binary)
plt.show()

print (prediction)


print("EOF")