import numpy as np
from Layers import Layer, FCLayer, ActivationLayer
from NN import Network
import ActivationFunction as AF
import LossFunction as LF 

import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
(train_images, train_labels), (test_images, test_labels) = mnist
train_images = train_images.reshape(train_images[0], 1, 28*28)
train_images = train_images.astype("float32")
train_images /= 255
train_labels = tf.one_hot(train_labels, depth=10)

test_images = test_images.reshape(test_images[0], 1, 28*28)
test_images = test_images.astype("float32")
test_images /= 255

#test_labels = tf.one_hot(test_labels, depth=10)



net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(AF.tanh, AF.tanh_prime))
net.add(FCLayer(100,50))
net.add(ActivationLayer(AF.tanh, AF.tanh_prime))
net.add(FCLayer(50,10))
net.add(ActivationLayer(AF.tanh, AF.tanh_prime))
net.use(LF.mse, LF.mse_prime)

net.fit(x_train=test_images, y_train=train_labels, epoches=20, learning_rate=0.01)

out = net.predict(test_images[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(test_labels[0:3])