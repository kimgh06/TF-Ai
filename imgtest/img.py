import tensorflow as tf
import matplotlib.pyplot as plt
(trX, trY), (teX, teY) = tf.keras.datasets.mnist.load_data()
print(trX.shape, trY)
(ctrX, ctrY), (cteX, cteY) = tf.keras.datasets.cifar10.load_data()
print(ctrX.shape, ctrY)
print(trY[:10])
print(ctrY[:10])
plt.imshow(trX[0], cmap='gray')
