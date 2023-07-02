import numpy as np
import matplotlib.pyplot as plt


x = np.array([[2, 1]])
w1 = np.array([[1, 2, 3],
            [4, 5, 6]])
b1 = np.array([[1, 2, 3]])
w2 = np.array([[3, 4, 0.5]])
b2 = 2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Dense(X, W, b):
    print(np.shape(X))
    print(np.shape(W))
    res = np.matmul(X, W)
    return sigmoid(res + b)

a1 = Dense(x, w1, b1)
a2 = Dense(a1, w2.T, b2)

print(a2)