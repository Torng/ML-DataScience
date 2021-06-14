import math
from Code.ML.deep_learning import Layer,tensor_apply,Tensor,tensor_combine,random_tensor
import numpy as np
from typing import List
from collections import Iterable
def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

class Sigmoid(Layer):
    def forward(self,input:Tensor):
        self.sigmoids = tensor_apply(sigmoid,input)
        return self.sigmoids
    def backward(self,gradient:Tensor):
        return tensor_combine(lambda sig,grad : sig*(1-sig)*grad,
                              self.sigmoids,
                              gradient)
class Linear(Layer):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 init:str='xavier'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = random_tensor(output_dim,input_dim,init=init)
        self.b = random_tensor(output_dim,init=init)

    def forward(self,input:Tensor):
        self.input = input
        return [np.dot(input,self.w[o])+self.b[o] for o in range(self.output_dim)]
    def backward(self,gradient:Tensor):
        self.b_grad = gradient
        self.w_grad = [[self.input[i]*gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]
        return [sum(self.w[o][i]*gradient[o] for o in range(self.output_dim)) for i in range(self.input_dim)]
    def params(self) ->Iterable[Tensor]:
        return [self.w,self.b]
    def grads(self) ->Iterable[Tensor]:
        return [self.w_grad,self.b_grad]

class Seqential(Layer):
    def __init__(self,layers:List[Layer]):
        self.layers = layers
    def forward(self,input:Tensor):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    def backward(self,gradient:Tensor):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    def params(self) ->Iterable[Tensor]:
        return (param for layer in self.layers for param in layer.params())
    def grads(self) ->Iterable[Tensor]:
        return (grad for layer in self.layers for grad in layer.grads())

