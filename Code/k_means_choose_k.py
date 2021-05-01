import numpy as np
from typing import List
from Code.k_means import KMeans,datas
from matplotlib import pyplot as plot
Vector = np.array

def squared_clustering_errors(inputs:List[Vector],k:int):
    model = KMeans(k)
    model.train(inputs)
    means = model.means

    assignments = [model.classify(input) for input in inputs]
    return sum(np.linalg.norm(input-means[cluster])
    for input,cluster in zip(inputs,assignments))

ks = range(1,len(datas)+1)

errors = [squared_clustering_errors(np.array(datas),k) for k in ks]

plot.plot(ks,errors)
plot.xticks(ks)

plot.ylabel("errors")
plot.show()
