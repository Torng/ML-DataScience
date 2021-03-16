
import random
import numpy as np
from scratch.linear_algebra import vector_mean,scalar_multiply,add

def linear_gradient(x:float,y:float,theta:float):
    slope,intercept = theta
    predicted = slope* x+intercept
    error = predicted-y
    square_error = error*2
    grad = [2*error*x,2*error]
    return grad
def gradient_step(v:np.array,gradient:np.array,step_size:float)->np.array:
    step = scalar_multiply(step_size,gradient)
    return add(v,step)
inputs = [(x,20*x+5) for x in range(-50,50)]

theta = [random.uniform(-1,1),random.uniform(-1,1)]

learn_rate = 0.001

for epoch in range(5000):
    grad = vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
    theta = gradient_step(theta,grad,-learn_rate)
    print(epoch,theta)

