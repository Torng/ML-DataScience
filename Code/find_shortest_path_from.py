from typing import Dict,List
import time


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

friendship = {user["id"]:[] for user in users}
for x,y in friendpair:
    friendship[x].append(y)
    friendship[y].append(x)



def shortest_path_from(id,friendship)->Dict[int,List[List[int]]]:
    shortest_path = {id:[[id]]}
    nows_friends_path = {id:friendship[id]}
    left_friends = list(friendship.keys())
    while(left_friends != []):
        next_neighbors = []
        for now,neighbors in nows_friends_path.items():
            next_neighbors += neighbors
            for neighbor in neighbors:
                # if(neighbor in shortest_path.keys()):
                #     continue
                # else:
                if(neighbor in shortest_path.keys()):
                    # shortest_path[neighbor].append(shortest_path[now]+[neighbor])
                    new_path = [s + [neighbor] for s in shortest_path[now]]
                    if(len(new_path[0])<len(shortest_path[neighbor][0])):
                        shortest_path[neighbor] = new_path
                    elif(len(new_path[0])==len(shortest_path[neighbor][0]) and new_path[0] not in shortest_path[neighbor]):
                        shortest_path[neighbor].append(new_path[0])
                    # shortest_path[neighbor].append(new_path)
                else:
                    shortest_path[neighbor] = [s+[neighbor] for s in shortest_path[now]]

        next_neighbors = list(set(next_neighbors))
        nows_friends_path = {next_neighbor:friendship[next_neighbor] for next_neighbor in next_neighbors}
        left_friends = list(set(left_friends)-set(shortest_path.keys()))
    return shortest_path
start = time.time()
shortest_path_from(3,friendship)
end = time.time()
print(round(end-start,6))





