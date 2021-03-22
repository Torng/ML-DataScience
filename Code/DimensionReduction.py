import numpy as np
from typing import List
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
Vector = np.array

def de_mean(data:List[Vector])->List[Vector]:
    mean = np.mean(data,axis=0)
    return [v-mean for v in data]

def vector_len(v:Vector)->float:
    return  math.sqrt(np.dot(v,v))

def direction(w:Vector)->Vector:
    w_len = vector_len(w)
    return [w_i/w_len for w_i in w]

def directional_varience(data:List[Vector],w:Vector)->float:
    w_dir = direction(w)
    return sum(np.dot(v,w_dir)**2 for v in data)

def directional_varience_gradient(data:List[Vector],w:Vector):
    w_dir = direction(w)
    return np.array([sum(2*np.dot(v,w_dir)*v[i] for v in data) for i in range(len(w))])

def first_principal_component(data:List[Vector],n:int,step_size:float=0.01)->Vector:
    guess = [1.0 for _ in data[0]]

    for i in range(n):
        dv = directional_varience(data,guess)
        gradient = directional_varience_gradient(data,guess)
        guess = guess+gradient*step_size
        print(i,guess)
    return direction(guess)


def project(v:Vector,w:Vector)->Vector:
    """v投影在w方向上得分量 w 長度為1"""
    w_dir = direction(w)
    projection_len = np.dot(v,w_dir)
    return Vector([w_i*projection_len for w_i in w])

def remove_projection_from_vector(v:Vector,w:Vector)->Vector:
    return v-project(v,w)
def remove_project(data:List[Vector],w:Vector)->List[Vector]:
    return [remove_projection_from_vector(v,w) for v in data]

def pca(data:List[Vector],num_components:int)->List[Vector]:
    components = []
    for _ in range(num_components):
        component = first_principal_component(data,100)
        components.append(component)
        data = remove_project(data,component)
    return components
def transform_vector(v:Vector,components:List[Vector])->Vector:
    return [np.dot(v,w) for w in components]
def transform(data:List[Vector],components:List[Vector])->Vector:
    return [transform_vector(v,components) for v in data]




