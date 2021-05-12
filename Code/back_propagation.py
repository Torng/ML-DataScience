import numpy as np
import math
from typing import List
from Code.NN_1 import feed_forward
import random
import tqdm

Vector = np.array
# 跳過
def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]
def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)
def sqerror_gradient(network:List[List[Vector]],inputs_vector:Vector,target_vector:Vector)->List[List[Vector]]:
    hidden_outputs , outputs = feed_forward(network,inputs_vector)

    output_deltas = [output*(1-output)*(output-target) for output,target in zip(outputs,target_vector)]

    output_grads = [ [output_deltas[i]*hidden_output for hidden_output in hidden_outputs] for i,output_neuron in enumerate(network[-1])]

    hidden_deltas = [hidden_output*(1-hidden_output*np.dot(output_deltas,[n[i] for n in network[i]]))
                     for i,hidden_output in enumerate(hidden_outputs)]
    hidden_grads = [[hidden_deltas[i]*input for input in inputs_vector+[1] ]for i,hidden_neuron in enumerate(network[0])]

    return [hidden_grads,output_grads]
network = [
    [
    [random.random() for _ in range(2+1)],
    [random.random() for _ in range(2+1)]
     ],
    [[random.random() for _ in range(2+1)]]
]
xs = [[0,0],[0,1],[1,0],[1,1]]
ys = [[0],[1],[1],[0]]

for epoch in tqdm.trange(20000,desc="neural net for XOR"):
    for x,y in zip(xs,ys):
        gradients = sqerror_gradient(network,x,y)
        network = [[gradient_step(neuron,grad,-1.0) for neuron,grad in zip(layer,layer_grad)]for layer,layer_grad in zip(network,gradients)]

