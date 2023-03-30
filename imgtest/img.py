import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
(trX, trY), (teX, teY) = tf.keras.datasets.mnist.load_data()
print(trX.shape, trY)
(ctrX, ctrY), (cteX, cteY) = tf.keras.datasets.cifar10.load_data()
print(ctrX.shape, ctrY)

# print(trY[0:10])
# plt.imshow(trX[0], cmap='gray')
# plt.show()
# print(trY[0])

# print(ctrY[0:10])
# plt.imshow(ctrX[0])
# plt.show()

dl = np.array([1, 2, 3, 4, 5])
print(dl.shape)

dl = np.array([dl, dl, dl, dl])
print(dl.shape)

dl = np.array([dl, dl, dl, dl, dl])
print(dl.shape)

x1 = np.array([1, 2, 3, 4, 5])
print(x1.shape)
print(trY[0:5])
print(trY[0:5].shape)

x2 = np.array([[1, 2, 3, 4, 5]])
print(x2.shape)

x3 = np.array([[1], [2], [3], [4], [5]])
print(x3.shape)
print(ctrY[0:5])
print(ctrY[0:5].shape)
