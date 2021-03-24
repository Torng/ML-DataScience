from collections import Counter
from typing import List,NamedTuple
import numpy as np
Vector = np.array()
def vote(labels:List[str])->str:
    vote_count = Counter(labels)
    winner,winner_count = vote_count.most_common(1)[0]
    num_winners = len([count for count in vote_count.values()
               if count==winner_count
               ])
    if num_winners==1:
        return winner
    else:
        return vote(labels[:-1])
class LabelPoint(NamedTuple):
    point:Vector
    label:str

def knn_classify(k:int,label_point:List[LabelPoint],new_point:Vector)->str:
    sort_by_distance = sorted(label_point,key= lambda lp:np.linalg.norm(lp.point - new_point))
    k_near_label = [lp.label for lp in sort_by_distance[:k]]
    return vote(k_near_label)
