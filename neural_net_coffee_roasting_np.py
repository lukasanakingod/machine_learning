"""Denna fil skippar kompileringssteget för modellen, modellen är vidare en sigmoid activation"""

import numpy as np
import lab_data.py as lab

x_train, y_train = lab.import_data()
W1_tmp = np.array([[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]])
b1_tmp = np.array([-9.82, -9.28,  0.96])
W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
b2_tmp = np.array([15.41])

def g(x):
    return 1/(1+np.exp(-x))

def np_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return(a_out)

def np_sequential(X, W1, b1, W2, b2):
    a1 = my_dense(X, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

def np_predict(X, W1, b1, W2, b2)
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = np_sequential(X, W1, b1, W2, b2)
    return(p)

X_tst = np.array([[200,13.9],  # postive example
                [200,17]])   # negative example
"""TODO fix normalization function X_tstn = norm_l(X_tst)  # remember to normalize"""
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

