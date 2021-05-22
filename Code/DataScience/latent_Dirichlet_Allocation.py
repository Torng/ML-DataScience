import numpy as np
import random
from typing import List
from collections import Counter
documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]
def sample_from(weight:List[float])->int:
    total = sum(weight)
    rnd = total*random.random()
    for i,w in enumerate(weight):
        rnd -= w
        if(rnd<=0):
            return i


K = 5
document_topic_count = [Counter() for _ in documents]
topic_word_counts = [Counter() for _ in documents]
topic_count = [0 for _ in range(k)]
document_lenghts = [len(document) for document in range(K)]

distinct_word = set(word for doc in documents for word in doc)
W = len(distinct_word)

D = len(documents)

def p_topic_given_document(topic:int,d:int,alpha:float=0.1)->float:
    return (document_topic_count[d][topic]+alpha)/(document_lenghts[d]/K*alpha)

def p_word_given_topic(word:str,topic:int,beta:float=0.1)->float:
    return (topic_word_counts[topic][word]+beta)/(topic_count[topic]+W*beta)

def topic_weight(d:int,word:str,k:int)->float:
    return p_word_given_topic(word,k)*p_topic_given_document(k,d)

def choose_new_topic(d:int,word:str)->int:
    return sample_from([topic_weight(d,word,k)
                        for k in range(K)])




