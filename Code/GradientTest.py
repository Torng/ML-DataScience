import matplotlib.pyplot as plot
import numpy as np
from sympy import diff, Symbol, sin, tan
import random
def add(v: np.array, w: np.array) -> np.array:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]
def scalar_multiply(c: float, v: np.array) -> np.array:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]
def gradient_step(v,gradient,step_size):
    step = scalar_multiply(step_size,v)
    return add(v,step)
# fx = [(x,x+100) for x in range(-10,10)]
fx = [(x,x+100) for x in range(-10,10)]
theta = np.array([random.uniform(-1,1),random.uniform(-1,1)])
# theta = [6,38]

def linear_gradient(x: float, y: float, theta: np.array):
    a,b = theta
    predicted = a*x+b
    error = predicted - y
    # grad = [2*x**4*a+2*b*x**2-2*x**2*y,2*a*x**2+2*b-2*y]
    grad = [2*x**2*a+2*b*x-2*x*y,2*a*x+2*b-2*y]
    # grad = [2*error*x , 2*error]
    return grad
learning_rate = 0.001
for epoch in range(10000):
    grad = np.mean([linear_gradient(x,y,theta) for x,y in fx],axis=0)

    grad_a,grad_b = grad
    a,b = theta
    theta = [a-grad_a*learning_rate,b-grad_b*learning_rate]
    print(epoch,theta)