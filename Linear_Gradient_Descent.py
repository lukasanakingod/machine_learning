import numpy as np
import matplotlib.pyplot as plt

# Skapar alla mätta x-värden (alltså bara x-värden 0-99)
x = np.linspace(0, 99, 100)

# Skapar alla mätta y-värden (Dessa är y-värden från 0-99, men sen så sprider jag de lite med hjälp av en random array)
y = np.linspace(0, 99, 100)
z = np.random.rand(100)*30
y = y + z

def compute_derivative_w(x, y, y_pred):
    n = len(y)
    dldw = np.dot((y_pred-y), x)
    return dldw/n

def compute_derivative_b(y, y_pred):
    n = len(y)
    dldb = sum(y_pred-y)
    return dldb/n

# Initialize boundary, learning_rate, and counter
boundary = 500
t = 0
learning_rate = 0.01

# Normalize x and y values to avoid large values
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)

# Initialize weight w and bias b
w = 0
b = 0

while t <= boundary:
    y_pred = w*x_normalized + b
    w -= learning_rate*compute_derivative_w(x, y_normalized, y_pred)
    b -= learning_rate*compute_derivative_b(y_normalized, y_pred)
    t += 1
    print(f"w: {w}, b: {b}")

# De-normalize the output data
w_denormalized = w * np.std(y) / np.std(x)
b_denormalized = (b * np.std(y)) - (w_denormalized * np.mean(x)) + np.mean(y) 

y_predicted =  w_denormalized*x + b_denormalized
plt.scatter(x, y)
plt.plot(x, y_predicted)
plt.show()
