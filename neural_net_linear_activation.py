import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )

a1 = linear_layer(X_train[0].reshape(1,1))

w, b= linear_layer.get_weights()

set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])

a1 = linear_layer(X_train[0].reshape(1,1))
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b

"""Calculate predictions tensorflow and numpy respectively"""
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b

print(prediction_np)

"""Plot"""
x = np.linspace(0, 2, 100)
plt.plot(X_train, prediction_tf)
plt.plot(X_train, prediction_np)
plt.scatter(X_train, Y_train)
plt.show()

"""
fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()
"""