import tensorflow as tf
import pandas as pd

(tx, ty), _ = tf.keras.datasets.cifar10.load_data()
ty = pd.get_dummies(ty.reshape(50000))
print(tx.shape, ty.shape)

X = tf.keras.layers.Input(shape=[32, 32, 3])
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.fit(tx, ty, epochs=10)
pred = model.predict(ty[0:5])
pd.DataFrame(pred).round(2)
ty[0:5]
model.summary()
