from collections import Counter,defaultdict
from typing import List,Tuple,Set,Dict
import numpy as np
import math

users_interests = [
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


def most_popular_new_interests(user_interests:List[str],
                               popular_interests:Counter,
                               max_result:int=5)->List[Tuple[str,int]]:
    suggestions = [(interest,frequency) for (interest,frequency) in popular_interests.most_common()
                   if interest not in users_interests]
    return suggestions[:max_result]

def make_user_interest_vector(user_interests:List[str],uniqe_interests:Set)->List[int]:
    return [1 if uniqe_interest in user_interests else 0 for uniqe_interest in uniqe_interests]

def cosine_similarity(v1:np.array , v2: np.array) -> float:
    return np.dot(v1, v2) / math.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

def most_similar_users_to(user_id:int,users_similarity:List[List[float]])->List[Tuple[int,float]]:
    pairs=[(other_user_id,similarity)
           for other_user_id,similarity in enumerate(users_similarity[user_id])
           if user_id != other_user_id and similarity>0]
    return sorted(pairs,key=lambda pair:pair[-1],reverse=True)

def user_based_suggestions(user_id:int,users_similarity:List[List[float]],include_current_interests:bool=False):
    suggestions:Dict[str,float] = defaultdict(float)
    for other_user_id,similarity in most_similar_users_to(user_id,users_similarity):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    suggestions = sorted(suggestions.items(),key=lambda pair:pair[-1],reverse=True)
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion,weight) for suggestion,weight in suggestions
                if suggestion not in users_interests[user_id]]

"""                            by item                            """

def most_similar_interests_to(interest_id:int,interest_similarities:np.array,interest_map):
    pairs = [(i,similarity) for i,similarity in enumerate(interest_similarities[interest_id])
           if i!=interest_id and similarity>0]
    pairs = [(interest_map[i],s) for i,s in pairs]
    return sorted(pairs,key=lambda pair:pair[-1],reverse=True)

def item_based_suggestions(user_id:int,users_interest_vectors:List[List[int]],interest_similarities:np.array,interest_map,include_current_interests:bool=False):
    suggestions:Dict[str,float] = defaultdict(float)

    user_interests = [i for i,is_interested in enumerate(users_interest_vectors[user_id])
                    if is_interested == 1]
    for users_interest in user_interests:
        for i,similarity in enumerate(interest_similarities[users_interest]):
            suggestions[interest_map[i]] += similarity

    suggestions = sorted(suggestions.items(),key=lambda pair:pair[-1],reverse=True)

    return suggestions


if __name__ == "__main__":
    users_interests_counts = Counter(interest for users_interest in users_interests for interest in users_interest)
    # print("users_interests_counts ------->", users_interests_counts)
    uniqe_interests = sorted({interest for users_interest in users_interests for interest in users_interest})
    # print("uniqe_interests------->", uniqe_interests)
    users_interest_vectors = [make_user_interest_vector(users_interest, uniqe_interests) for users_interest in
                             users_interests]
    # print(most_popular_new_interests(users_interests[0], users_interests_counts))
    # print("users_interest_vector------->", users_interest_vectors)

    users_similarity = [[cosine_similarity(vector_i, vector_j)
                         for vector_j in users_interest_vectors]
                        for vector_i in users_interest_vectors]
    # print("users_similarity---->", users_similarity)
    # print(most_similar_users_to(0, users_similarity))
    # print(user_based_suggestions(0, users_similarity))

    """                            by item                            """

    interest_user_matrix = np.array(users_interest_vectors).T
    print("interest_user_matrix",interest_user_matrix)
    interest_similarities = [[cosine_similarity(interest_user_i,interest_user_j) for interest_user_j in interest_user_matrix]for interest_user_i in interest_user_matrix]
    # print(np.array(interest_similarities)[0])
    # print(most_similar_interests_to(0,interest_similarities,uniqe_interests))

    print(item_based_suggestions(0,users_interest_vectors,interest_similarities,uniqe_interests))