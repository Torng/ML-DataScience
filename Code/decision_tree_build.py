from typing import NamedTuple,Union,Any,List
from collections import Counter
from Code.decision_tree import partition_entropy_by,partition_by,candidates,Candidate
class Leaf(NamedTuple):
    value:Any
class Split(NamedTuple):
    attribute:str
    subtrees:dict
    default_value:Any = None
DecisionTree = Union[Leaf,Split]

def classify(tree:DecisionTree,input:Any):
    if isinstance(tree,Leaf):
        return tree.value
    subtree_key = getattr(input,tree.attribute)

    if(subtree_key not in tree.subtrees):
        return tree.default_value
    subtree = tree.subtrees[subtree_key]

    return classify(subtree,input)

def build_tree_id3(inputs:List[Any],split_attributes:str,target_attribute:str)->DecisionTree:
    label_count = Counter(getattr(input,target_attribute)
                          for input in inputs)
    most_common_label = label_count.most_common(1)[0][0]
    if(len(label_count)==1 or not split_attributes):
        return Leaf(most_common_label)
    def split_entropy(attribute:str)->float:
        return partition_entropy_by(inputs,attribute,target_attribute)
    best_attribute = min(split_attributes,key=split_entropy)

    partitions = partition_by(inputs,best_attribute)

    new_attribute = [a for a in split_attributes if a != best_attribute]

    subtrees = {attribute_value:build_tree_id3(subset,new_attribute,target_attribute)
                for attribute_value,subset in partitions.items()}
    return Split(best_attribute,subtrees,most_common_label)

tree = build_tree_id3(candidates,['level','lang','tweets','phd'],'did_well')
# print(tree)

print(classify(tree,Candidate("Junoir","Python",True,False)))


