import pandas as pd
from typing import NamedTuple,List,Dict
import random
import scipy.stats as st
import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import json
class Rating(NamedTuple):
    user_id:str
    movie_id:str
    rating:float
MOVIES = "../../Data/movies/u.item"
RATING = "../../Data/movies/u.data"
def write_dict_to_json(dict:Dict,output_name:str)->None:
    dict = {int(key):list(val.astype(float)) for key,val in dict.items()}
    with open(output_name+'.txt', 'w') as file:
        file.write(json.dumps(dict))
def read_json_to_dict(file:str)->Dict:
    with open(file+'.txt') as f:
        return json.load(f)
def random_tensor(dim:int)->np.array:
    return np.array([st.norm.ppf(random.random()) for _ in range(dim)])

def train(dataset:List[Rating],
          movie_vectors:Dict[str,List[float]],
          user_vectors:Dict[str,List[float]],
          learning_rate:float=None)->None:
    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i,rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = np.dot(user_vector,movie_vector)
            error = predicted - rating.rating
            loss += error**2


            if( learning_rate is not None):
                user_gradient = [error * m_j for m_j in movie_vector]
                movie_gradient = [error * u_j for u_j in user_vector]

                for j in range(EMBEDDING_DIM):
                    user_vector[j] -= learning_rate*user_gradient[j]
                    movie_vector[j] -= learning_rate*movie_gradient[j]
                t.set_description(f"avg loss :{loss/(i+1)}")
movies_df = pd.read_csv(MOVIES,delimiter="|",encoding='iso-8859-1',header=None)
rating_df = pd.read_csv(RATING,delimiter="\t",header=None)
movies = {movie_row[0]:movie_row[1] for i,movie_row in movies_df.iterrows()}
rating = [Rating(rating_row[0],rating_row[1],rating_row[2]) for i,rating_row in rating_df.iterrows()]


random.seed(0)
random.shuffle(rating)
split1 = int(len(rating)*0.7)
split2 = int(len(rating)*0.85)
train_data = rating[:split1]
validation = rating[split1:split2]
test = rating[split2:]


avg_rating = sum(rate.rating for rate in train_data)/len(train_data)
baseline_error = sum((rate.rating-avg_rating)**2 for rate in test)/len(test)

user_ids = {rate.user_id for rate in rating}
movie_ids = {rate.movie_id for rate in rating}
EMBEDDING_DIM = 2
user_vectors ={user_id:random_tensor(EMBEDDING_DIM) for user_id in user_ids}
movie_vectors ={movie_id:random_tensor(EMBEDDING_DIM) for movie_id in movie_ids}
print(user_vectors)
learning_rate = 0.05
for epoch in range(20):
    learning_rate *= 0.9
    print(epoch,learning_rate)
    train(train_data,movie_vectors,user_vectors,learning_rate)
train(test,movie_vectors,user_vectors)

write_dict_to_json(movie_vectors,"movie_vectors")
write_dict_to_json(user_vectors,"user_vectors")

original_vector = [vector for vector in movie_vectors.values()]
vector_df = pd.DataFrame(data=original_vector,columns=["x","y"])
sns.scatterplot(x="x",y="y",data=vector_df)
plt.show()

pca = PCA(n_components=2)
pca.fit(original_vector)
X_pca = pca.transform(original_vector)
X_new = pca.inverse_transform(X_pca)

vector_df["x_new"] = X_new[:, 0]
vector_df["y_new"] = X_new[:, 1]
print(X_new)

sns.scatterplot(x="x_new",y="y_new",data=vector_df)
plt.show()




