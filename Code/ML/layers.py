import math
from Code.ML.deep_learning import Layer,tensor_apply,Tensor,tensor_combine,random_tensor,Loss,Optimizer
import numpy as np
from typing import List
from collections import Iterable
def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))
def tanh(x:float)->float:
    if(x<-100):
        return -1
    elif(x>100):
        return  1
    em2x = math.exp(-2*x)
    return (1-em2x)/(1+em2x)

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
        return [sum(self.w[o][i]*gradient[o]
                    for o in range(self.output_dim))
                for i in range(self.input_dim)]
    def params(self) :
        return [self.w,self.b]
    def grads(self) :
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
    def params(self) :
        return (param for layer in self.layers for param in layer.params())
    def grads(self) :
        return (grad for layer in self.layers for grad in layer.grads())

class SSE(Loss):
    def loss(self,predicted:Tensor,actual:Tensor):
        square_error =  tensor_combine(lambda predicted, actual:(predicted-actual)**2,
                              predicted,
                              actual)
        return  np.sum(square_error)
    def gradient(self,predicted:Tensor,actual:Tensor):
        return tensor_combine(lambda predicted, actual:2*(predicted-actual),
                              predicted,
                              actual)

class Momentum(Optimizer):
    def __init__(self,learning_rate:float,momentum:float=0.9):
        self.lr = learning_rate
        self.mo = momentum
        self.updates:List[Tensor]=[]
    def step(self,layer:Layer) ->None:
        if not self.updates:
            self.updates = [np.zeros_like(grad) for grad in layer.grads()]
        for update,param,grad in zip(self.updates,
                                     layer.params(),
                                     layer.grads()):
            update[:] = tensor_combine(lambda u,g : self.mo*u+(1-self.mo)*g,
                                       update,
                                       grad)
            param[:] = tensor_combine(lambda p,u : p-self.lr*u,
                                      param,
                                      update)

class GradientDescent(Optimizer):
    def __init__(self,learning_rate:float=0.1):
        self.lr = learning_rate

    def step(self,layer:Layer) ->None:
        for param,grad in zip(layer.params(),layer.grads()):
            param[:] = tensor_combine(lambda param,grad:param-grad*self.lr,
                                      param,
                                      grad)
class Tanh(Layer):
    def forward(self,input:Tensor):
        self.tanh = tensor_apply(tanh,input)
        return self.tanh
    def backward(self,gradient:Tensor):
        return tensor_combine(lambda tanh,grad : (1-tanh**2)*grad,
                              self.tanh,
                              gradient)

class Relu(Layer):
    def forward(self,input:Tensor):
        self.input = input
        return tensor_apply(lambda x: max(x,0),input)
    def backward(self,gradient:Tensor):
        return tensor_combine(lambda x,grad: grad if x>0 else 0,
                              self.input,
                              gradient)
