import numpy as np
from typing import List
import itertools
from Code.DataScience.k_means import datas
from matplotlib import pyplot as plt
import random

class KMeans:
    def __init__(self, k: int):
        self.k = k
        self.means = np.array

    def _classify_by_distance(self, vector: np.array) -> int:
        return min(range(self.k), key=lambda i: np.linalg.norm(np.array(vector) - self.means[i]))

    def _recalculate(self, inputs: List[np.array]):
        return [self._classify_by_distance(input) for input in inputs]

    def _reset_means(self,inputs:List[np.array],assignments:List[int]):
        clusters = [[] for _ in range(self.k)]
        for assignment,input in zip(assignments,inputs):
            clusters[assignment].append(input)

        return [np.mean(cluster,axis=0) if cluster else random.choice(inputs) for cluster in clusters ]
    def _diffence(self,assigment,new_assigment):
        return len([x_i for x_i,y_i in zip(assigment,new_assigment) if x_i!=y_i])
    def train(self, inputs: List[np.array]):
        self.means = np.random.randn(self.k, len(inputs[0]))
        old_assignments = [random.randint(0,self.k) for _ in range(len(inputs))]
        for _ in itertools.count():
            new_assignments = self._recalculate(inputs)
            if(self._diffence(old_assignments,new_assignments)==0):
                return
            old_assignments = new_assignments
            self.means = self._reset_means(inputs,old_assignments)

model = KMeans(2)
model.train(datas)
x =[data[0] for data in datas]
y =[data[1] for data in datas]
print(model.means)
x_m = [mean[0] for mean in model.means]
y_m = [mean[1] for mean in model.means]
plt.scatter(x,y)
plt.scatter(x_m,y_m,color="red")
plt.show()



