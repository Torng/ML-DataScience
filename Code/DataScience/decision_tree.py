from collections import  Counter,defaultdict
from typing import List,Any,NamedTuple,Optional,TypeVar,Dict
import math

def entropy(class_probabilities:List[float]):
    return sum(-prob*math.log(prob,2) for prob in class_probabilities
               if prob >0)
def class_probabilities(labels:List[Any]):
    total_count = len(labels)
    return [count/total_count for count in Counter(labels).values()]
def data_entropy(labels:List[Any])->float:
    return entropy(class_probabilities(labels))
def partition_entropy(subsets:List[List[Any]])->float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset)*len(subset)/total_count for subset in subsets)
class Candidate(NamedTuple):
    level:str
    lang:str
    tweets:bool
    phd:bool
    did_well:Optional[bool] = None
datas = [
        ('Senior','Java',False,False,   False),
        ('Senior','Java',False,True,  False),
        ('Mid','Python',False,False,     True),
        ('Junior','Python',False,False,  True),
        ('Junior','R',True,False,      True),
        ('Junior','R',True,True,    False),
        ('Mid','R',True,True,        True),
        ('Senior','Python',False,False, False),
        ('Senior','R',True,False,      True),
        ('Junior','Python',True,False, True),
        ('Senior','Python',True,True,True),
        ('Mid','Python',False,True,    True),
        ('Mid','Java',True,False,      True),
        ('Junior','Python',False,True,False)
    ]

candidates = [Candidate(level=data[0],
                        lang=data[1],
                        tweets=data[2],
                        phd=data[3],
                        did_well = data[4]) for data in datas]

T = TypeVar('T')

def partition_by(inputs:List[T],attritbute:str)->Dict[Any,List[T]]:
    partition:Dict[Any,List[T]]= defaultdict(list)
    for input in inputs:
        key = getattr(input,attritbute)
        partition[key].append(input)
    return partition
def partition_entropy_by(inputs:List[Any],attritbute:str,label_attribute)->float:
    partitions = partition_by(inputs,attritbute)
    labels = [[getattr(inputs,label_attribute) for inputs in partition] for partition in partitions.values()]
    return partition_entropy(labels)


