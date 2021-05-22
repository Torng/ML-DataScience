import numpy as np
from typing import NamedTuple,List,Tuple

Vector = np.array
inputs = np.array([[63,160,150],[67,170.2,160],[70,177.8,171]])


def scale(data:List[Vector])->Tuple[Vector,Vector]:
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return mean,std


print(scale(inputs))
def rescale(data:List[Vector])->List[Vector]:
    mean ,std = scale(data)
    dim = len(data[0])
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if std[i]>0:
                v[i]=(v[i]-mean[i])/std[i]
    return rescaled
print(scale(rescale(inputs)))
