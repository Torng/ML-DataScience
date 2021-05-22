

import numpy as np

def least_squares(x,y):
    """least_squares by linear algebra"""
    n = len(y)
    x_mat = np.c_[np.ones(n), x]
    a = x_mat.T.dot(x_mat)
    b = x_mat.T.dot(y)
    return np.linalg.solve(a,b)
def ridge_least_squares(y,x_mat,alpha):
    a = x_mat.T.dot(x_mat) +alpha*np.identity(x_mat.T.dot(x_mat).shape[0])
    b = x_mat.T.dot(y)
    return np.linalg.solve(a,b)
# def least_squares_by_gradient():


