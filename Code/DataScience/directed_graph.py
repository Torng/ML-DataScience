from collections import Counter
from typing import Tuple,List,Dict
import networkx as nx
from matplotlib import pyplot as plt
import tqdm
users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]
endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
                (2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
                (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

endorsement_count = Counter(target for source,target in endorsements)

def draw_basic_network_graph(nodes):
    G = nx.DiGraph()
    G.add_edges_from(nodes)
    G.size()
    plt.figure(figsize=(8,6))
    nx.draw(G, with_labels=True, arrows=True,node_size=4000, font_size=20)
    plt.draw()
    plt.show()
def page_rank(users:List[Dict],
              endorsements:List[Tuple[int,int]],
              damping:float=0.85,
              num_iters:int=100)->Dict[int,float]:
    outgoing_count = Counter(target for source,target in endorsements)
    num_users = len(users)
    pr = {user["id"]:1/num_users for user in users}
    base_pr = (1-damping)/num_users
    for iter in tqdm.trange(num_iters):
        next_pr = {user["id"]:base_pr for user in users}
        for source,target in endorsements:
            next_pr[target] += damping *pr[source] / outgoing_count[source]
        pr = next_pr
    return pr
draw_basic_network_graph(endorsements)
pr = page_rank(users,endorsements)
print(pr)

