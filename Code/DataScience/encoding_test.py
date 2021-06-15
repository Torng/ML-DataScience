import numpy as np
import pandas as pd
from typing import List

def label_encoding(labels:List[str]):
    labels_dict = {}
    index = 0
    ans_label = []
    for label in labels:
        if(label not in labels_dict.keys()):
            labels_dict[label] = index
            ans_label.append(index)
            index += 1
        else:
            ans_label.append(labels_dict[label])
    return ans_label

def onehot_encoding(labels:List[str]):

    ans_labels = []
    labels_dict = {}
    index = 0
    for label in labels:
        init_arr = np.zeros((1, len(set(labels))))
        if(label not in labels_dict.keys()):

            init_arr[0,index]=1
            ans_labels.append(init_arr)
            labels_dict[label] = init_arr
            index += 1
        else:
            ans_labels.append(labels_dict[label])
    return ans_labels
if(__name__=="__main__"):
    country = ['Taiwan', 'Australia', 'Ireland', 'Australia', 'Ireland', 'Taiwan']
    age = [25, 30, 45, 35, 22, 36]
    salary = [20000, 32000, 59000, 60000, 43000, 52000]
    dic = {'Country': country, 'Age': age, 'Salary': salary}
    data = pd.DataFrame(dic)
    print(onehot_encoding(country))