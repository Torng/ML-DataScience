from collections import Counter,defaultdict
from typing import List,Tuple,Set,Dict
import numpy as np
import math
from Code.DataScience.recommender_systems import users_interests

users_interests_T = np.array(users_interests).T
print(users_interests_T)