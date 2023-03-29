import tensorflow as tf
import matplotlib.pyplot as plt
(trX, trY), (teX, teY) = tf.keras.datasets.fashion_mnist.load_data()
print(trX.shape, trY)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankleboot']
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                           activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])
model.fit(trX, trY, epochs=10)

trX.reshape((trX.shape[0], trX.shape[1], trX.shape[2], 1))
teX.reshape((teX.shape[0], teX.shape[1], teX.shape[2], 1))
