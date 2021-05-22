import numpy as np
import math
from typing import List
Vector = np.array
def step_function(x:float)->float:
    return 1.0 if x >= 0 else 0.0

def perceptrop_out(weight:Vector,bias:float,x:Vector):
    calculation = np.dot(weight,x)+bias
    return step_function(calculation)

def sigmoid(x:float)->float:
    return 1/(1+math.exp(-x))

def neuron_output(weights:Vector,inputs:Vector)->float:
    return sigmoid(np.dot(weights,inputs))

def feed_forward(neural_network:List[List[Vector]],inputs:Vector)->List[Vector]:
    outputs:List[Vector] = []
    input_vector = inputs
    for layer in neural_network:
        # input_with_bias = inputs.append([1])
        input_with_bias = np.append(input_vector,[1])
        output = [neuron_output(neuron,input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output.copy()
    return outputs

xor_network = [
    [
        [20,20,-30],
        [20,20,-10]
    ],
    [
        [-60,60,-30]
     ]
]
print(feed_forward(xor_network,np.array([1,1])))