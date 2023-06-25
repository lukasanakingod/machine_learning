import numpy as np
import matplotlib.pyplot as plt


# Skapar alla mätta x-värden (alltså bara x-värden 0-99)
X1 = np.linspace(0, 99, 100)
X2 = np.random.uniform(low=-1, high=1, size=100)
X3 = np.random.uniform(low=0, high=10, size=100)


X = np.column_stack((X1, X2, X3))


# Skapar alla mätta y-värden (Dessa är y-värden från 0-99, men sen så sprider jag de lite med hjälp av en random array)
y = np.linspace(0, 99, 100)
z = np.random.rand(100)*30
y = y + z

def compute_derivative_w_i(X, y, y_pred):
    n = len(y)
    dldw = np.dot(X.T, y_pred-y)
    return dldw/n

def compute_derivative_b(y, y_pred):
    n = len(y)
    dldb = sum(y_pred-y)
    return dldb/n

# Initialize boundary, learning_rate, and counter
boundary = 500
t = 0
learning_rate = 0.01

x_normalized = (X - np.mean(X)) / np.std(X)
y_normalized = (y - np.mean(y)) / np.std(y)

print(x_normalized)

# Initialize weight w and bias b
num_features = x_normalized.shape[1]
w = np.zeros(num_features)
b = 0

while t <= boundary:
    y_pred = np.dot(x_normalized, w) + b
    w -= learning_rate*compute_derivative_w_i(x_normalized, y_normalized, y_pred)
    b -= learning_rate*compute_derivative_b(y_normalized, y_pred)
    t += 1
    print(f"w: {w}, b: {b}")

w_denormalized = w * np.std(y) / np.std(X)
b_denormalized = (b * np.std(y)) - np.sum(w_denormalized * np.mean(X)) + np.mean(y)
y_predicted = np.dot(X, w_denormalized) + b_denormalized


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter_actual = ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual')
scatter_pred = ax.scatter(X[:, 0], X[:, 1], y_predicted, color='red', label='Predicted')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Y')

xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=10),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), num=10))
zz = w_denormalized[0] * xx + w_denormalized[1] * yy + b_denormalized
ax.plot_surface(xx, yy, zz, alpha=0.3, color='g')

plt.show()
