import numpy as np
from typing import List
import random
import itertools
from matplotlib import pyplot as plot
Vector = np.array
datas = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
x = [data[0] for data in datas]
y = [data[1] for data in datas]
plot.scatter(x,y)
def num_difference(v1:Vector,v2:Vector)->int:
    return len([ x1 for x1,x2 in zip(v1,v2) if x1!=x2])

def cluster_means(k:int,inputs:List[Vector],assignments:List[int]) -> List[Vector]:
    clusters = [[] for _ in range(k)]
    for input,assignment in zip(inputs,assignments):
        clusters[assignment].append(input)
    return [ np.mean(cluster,axis=0) if cluster else random.choice(inputs)
        for cluster in clusters]
class KMeans:
    def __init__(self,k:int):
        self.k = k
        self.means = None
    def classify(self,input:Vector):
        return min(range(self.k),
                   key=lambda i:np.linalg.norm(input-self.means[i]))
    def train(self,inputs:List[Vector])->None:
        assignments = [random.randrange(self.k) for _ in inputs]

        for _ in itertools.count():
            self.means = cluster_means(self.k,inputs,assignments)
            new_assignments = [self.classify(input) for input in inputs]
            num_changed = num_difference(assignments,new_assignments)
            if num_changed == 0:
                return
            assignments = new_assignments
            self.means = cluster_means(self.k,inputs,assignments)

model = KMeans(3)
model.train(np.array(datas))
print(model.means)
cluster_x = [x_i[0] for x_i in model.means]
cluster_y = [y_i[1] for y_i in model.means]

plot.scatter(cluster_x,cluster_y,color="red")
plot.show()

