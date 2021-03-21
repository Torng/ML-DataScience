import numpy as np
from typing import List
import math
import matplotlib.pyplot as plt
import random
x = [1,2,3,4,5,6,7]
y = [1,1.5,2,3,4.5,5,6.7]
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
        guess = guess-gradient*step_size
        print(i,guess)
    return direction(guess)

datas = list(zip(x,y))


x_new,y_new = zip(*de_mean(datas))
print(x_new,y_new)
# plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(x_new, y_new)
# plt.xticks([-2,-1,0,1,2,3,4,5]) #設定x軸刻度
# plt.yticks([-2,-1,0,1,2,3,4,5])
plt.xlim(-5,6)
plt.ylim(-3,3)
v_1,v_2 = first_principal_component(de_mean(datas),1000)

plt.quiver(0,0,v_1,v_2, color=['r'], scale=10)
plt.show()




