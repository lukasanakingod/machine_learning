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
    dldw = sum((y_pred-y)*x)
    return dldw/n

def compute_derivative_b(x, y, y_pred):
    n = len(y)
    dldb = sum(y_pred-y)
    return dldb/n

# Initialize boundary, learning_rate, and counter
boundary = 500
t = 0
learning_rate = 0.01

# Initialize weight w and bias b
w = 0
b = 0

while t <= boundary:
    y_pred = w*x + b
    w = w - learning_rate*compute_derivative_w(x, y, y_pred)
    b = b - learning_rate*compute_derivative_b(x, y, y_pred)
    t += 1
    print(f"w: {w}, b: {b}")

y_predicted =  w*x + b
plt.scatter(x, y)
plt.plot(x, y_predicted)
plt.show()
