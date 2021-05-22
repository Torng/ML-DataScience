import numpy as np
from typing import Tuple

from Code.DataScience.DimensionReduction import de_mean
Vector = np.array()
def predice(alpha:float,beta:float,x_i:float)->float:
    return beta*x_i+alpha
def error(alpha:float,beta:float,x_i:float,y_i:float)->float:
    return predice(alpha,beta,x_i)-y_i

def sum_of_sqerror(alpha:float,beta:float,x:Vector,y:Vector)->float:
    return sum(error(alpha,beta,x_i,y_i)**2 for x_i,y_i in zip(z,y))
def total_sum_of_squares(y:Vector)->float:
    return sum(v ** 2 for v in de_mean(y))
def r_squared(alpha:float,beta:float,x:Vector,y:Vector):
    return 1.0 - (sum_of_sqerror(alpha,beta,x,y)/total_sum_of_squares(y))
def least_squares_fit(x:Vector,y:Vector)->Tuple[float,float]:




x = [i for i in range(0,100,10)]
y = [2*x_i+1 for x_i in x]
