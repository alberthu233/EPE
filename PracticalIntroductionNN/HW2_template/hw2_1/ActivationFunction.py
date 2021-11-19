import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def relu(X):
   return np.maximum(0,X)

def relu_prime(x):
    if x > 0:
        return 1
    else:
        return 0

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum