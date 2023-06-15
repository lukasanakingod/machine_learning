import numpy as np
from matplotlib import pyplot as plt

# Skapar x-värden i form av temperaturer från 0-50 (51 datapunkter)
temperature = np.linspace(0, 50, 51)

#Skapar en array med 25 ettor och 26 nollor. Detta indikerar att allting på index 0-24 (korresponderar till temperaturer) 
#och 24-50 är överlevbart respektive icke-överlevbart. Detta är våra y-värden
survivability = np.append(np.ones(25), np.zeros(26))

"""Detta är logistic sigmoid funktionen som är på formen 1/(1+e^-x). Den tar in två vektorer och dottar dessa"""
def sigmoid(theta, x):
    return 1 / (1 + np.exp(-np.dot(theta, x)))

"""Denna funktion utför beräkningen av derivatan. Se 2.3 Gradient Optimization Defining The Loss"""
def partial_derivative(Y, X, theta):
    n = len(Y)
    X_with_bias = np.column_stack((np.ones(n), X))  # Add a column of ones to X
    d_theta = -np.dot(X_with_bias.T, (Y - sigmoid(theta, X_with_bias))) / n # Funkar för man utför sigmoid på hela x_with_bias mha numpy
    return d_theta

"""Beräknar hur mycket vi förändrar theta-vektorn enligt step-metod. Se 2.3 Gradient Optimization Defining The Loss"""
def descent():
    theta = np.array([0.01, 0.01])  # Initialiserar theta-vektorn med två godtyckliga värden. 
    dTheta = np.zeros(len(theta)) # Skapar en vektor som innehåller nollor. Denna kommer användas för att uppdatera theta-värdena
    boundary = 10000 # Antalet uppdateringssteg vi tar
    for t in range(boundary):
        dTheta = 0.1 * partial_derivative(survivability, temperature, theta)
        theta -= dTheta
    return theta

# Plottar vår sigmoid graf mot alla observerade datapunkter.
theta_obs = descent()
plt.scatter(temperature, survivability, label="temperature")
plt.xlabel('Temperature')
plt.ylabel('Survivability')
plt.title('Sigmoid Function')
plt.legend()

plt.plot(temperature, sigmoid(theta_obs, np.column_stack((np.ones(len(temperature)), temperature))), color="red")  # Plot the final sigmoid

plt.show()