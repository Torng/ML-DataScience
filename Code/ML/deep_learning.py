from typing import Callable,Iterable
import numpy as np
Tensor = np.array

def tensor_apply(f:Callable[[float],float],tensor:Tensor):
    if np.array(tensor).ndim==1:
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f,tensor_i) for tensor_i in tensor]

def tensor_combine(f:Callable[[float,float],float],
                    t1:Tensor,
                    t2:Tensor)->Tensor:
    if(np.array(t1).ndim==1):
        return [f(x,y) for x,y in zip(t1,t2)]
    else:
        return [tensor_combine(f,t1_1,t2_1) for t1_1,t2_1 in zip(t1,t2)]

def random_tensor(*dims:int,init:str='normal'):
    if(init=='normal'):

        return np.random.standard_normal(size=dims)
    elif(init=="uniform"):
        return np.random.uniform(size=dims)
    elif(init=="xavier"):
        std = np.sqrt(2. / (len(dims) + sum(dims)))
        return np.random.normal(loc=0., scale=std, size=dims)
    else:
        raise ValueError(f"unknow init : {init}")
class Layer:
    def forward(self,input:Tensor):
        raise NotImplementedError

    def backward(self,gradient:Tensor):
        raise NotImplementedError

    def params(self):
        return ()

    def grads(self):
        return ()

class Loss:
    def loss(self,predicted:Tensor,actual:Tensor):
        raise NotImplementedError
    def gradient(self,predicted:Tensor,actual:Tensor):
        raise NotImplementedError

class Optimizer():
    def step(self,layer:Layer)->None:
        raise NotImplementedError