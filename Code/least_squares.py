

import numpy as np

def least_squares(y,x_mat):
    a = x_mat.T.dot(x_mat)
    b = x_mat.T.dot(y)
    return np.linalg.solve(a,b)
def ridge_least_squares(y,x_mat,alpha):
    a = x_mat.T.dot(x_mat) +alpha*np.identity(x_mat.T.dot(x_mat).shape[0])
    b = x_mat.T.dot(y)
    return np.linalg.solve(a,b)
x = [x_i for x_i in range(-10,10)]
y = [x_i*4 for x_i in x]

n = len(y)
x_mat = np.c_[np.ones(n),x]
# ans_arr = least_squares(y,x_mat)
ans_arr = ridge_least_squares(y,x_mat,0.1)
print(ans_arr)
