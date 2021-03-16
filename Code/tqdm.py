import tqdm
import random
with tqdm.trange(3,100) as t :
    for i in t:
        _ = [random.random() for _ in range(1000000)]