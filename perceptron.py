import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
DATA = np.array([[2.0, 3.5, 1,1],
                 [3.0, 2.5, 1,1],
                 [3.5, 4.0, 1,1],
                 [4.5, 1.5, 1,1],
                 [5.0, 3.0, 1,1],
                 [5.5, 2.0, 1,1],
                 [1.0, 1.0, 1,-1],
                 [2.0, 1.5, 1,-1],
                 [2.5, 0.5, 1,-1],
                 [3.0, 0.5, 1,-1],
                 [3.5, 1.0, 1,-1],
                 [4.0, 0.5, 1,-1]])
print(DATA[0][len(DATA[0])-1])
X = np.linspace(-10,10,100)
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def perceptron(samples,boundary,b):
    k = 0
    i = 0
    data_len = len(samples[0])
    theta_obs = np.zeros(data_len-1)
    print(theta_obs)
    theta_obs[data_len-2] = b

    while k < boundary:
        attempts = [0,10,100,1000]
        x_i = samples[i][0:data_len-1]
        y_i = samples[i][data_len-1]
        y_i_obs = sign(np.dot(theta_obs,x_i))
        if y_i != y_i_obs:
            theta_obs = theta_obs + y_i*x_i
        k = k+1
        if(k in attempts):
            Y = -(theta_obs[0] / theta_obs[1]) * X - (theta_obs[2] / theta_obs[1])
            plt.plot(X,Y)
            print("LOL")
        if i+1 >= len(samples):
            i = 0
        else:
            i+=1
        print("Theta obs: ",theta_obs," Data point: ",x_i, " Y point: ",y_i, " Y obs: ", y_i_obs,"i: ",i," ", "\n")
        Y = -(theta_obs[0] / theta_obs[1]) * X - (theta_obs[2] / theta_obs[1])
    return (theta_obs,Y)
# Define the colormap colors
colors = ['red', 'green']

# Define the corresponding color map positions
bounds = [-1, 1]

# Create the colormap object
cmap = ListedColormap(colors)

# Set the colormap boundaries
norm = plt.Normalize(bounds[0], bounds[1])
collection = perceptron(DATA,10000,1)
plt.plot(X,collection[1])
plt.scatter(DATA[:,0],DATA[:,1],c=DATA[:,3],cmap=cmap)
plt.plot()
plt.show()


