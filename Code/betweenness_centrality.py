from typing import Dict,List
from collections import deque
import time
from matplotlib import pyplot as plt
import networkx as nx
Path = List[int]
Friendships =Dict[int,List[int]]

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
friendpair = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
friendships:Friendships = {user["id"]:[] for user in users}
for i,j in friendpair:
    friendships[i].append(j)
    friendships[j].append(i)
def shortest_paths_from(from_user_id:int,friendships:Friendships)->Dict[int,List[Path]]:
    shortest_paths_to:Dict[int,List[Path]] = {from_user_id:[[]]}
    frontier = deque((from_user_id,friend_id)for friend_id in friendships[from_user_id])
    while frontier:
        prev_user_id,user_id = frontier.popleft()
        paths_to_prev_user = shortest_paths_to[prev_user_id]
        new_paths_to_user = [path+[user_id] for path in paths_to_prev_user]
        old_paths_to_user = shortest_paths_to.get(user_id,[])

        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')
        new_paths_to_user = [path for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]
        shortest_paths_to[user_id] = old_paths_to_user+new_paths_to_user

        frontier.extend((user_id,friend_id)
                        for friend_id in friendships[user_id]
                        if friend_id not in shortest_paths_to)
    return shortest_paths_to
shortest_paths = {user["id"]:shortest_paths_from(user["id"],friendships)
                  for user in users}
betweenness_centrlity = {user["id"]:0.0 for user in users}
for source in users:
    for target_id , paths, in shortest_paths[source["id"]].items():
        if(source["id"]<target_id):
            num_paths = len(paths)
            contrib = 1/num_paths
            for path in paths:
                for between_id in path:
                    if between_id not in [source["id"],target_id]:
                        betweenness_centrlity[between_id] +=contrib
def draw_basic_network_graph(nodes,node_sizes):
    G = nx.Graph()
    G.add_edges_from(nodes)
    plt.figure(figsize=(8,6))
    nx.draw(G, with_labels=True, node_size=node_sizes, font_size=20, font_family='Source Han Sans TW',font_color="yellow", font_weight="bold")
    plt.show()

print(betweenness_centrlity)
node_sizes = [500 if value == 0.0 else value*500for between_id,value in betweenness_centrlity.items() ]
draw_basic_network_graph(friendpair,node_sizes)




