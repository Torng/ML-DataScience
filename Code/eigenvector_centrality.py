import numpy as np
from Code.betweenness_centrality import draw_basic_network_graph


friendpair = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
def make_adjacency_matrix(n,relations)->np.array:
    zero_matrix = np.zeros((n,n))
    # for i in range(len(zero_matrix)):
    #     zero_matrix[i][i] = 1
    for relation in relations:
        i,j = relation
        zero_matrix[i][j] = 1
        zero_matrix[j][i] = 1
    return zero_matrix




adjacency_matrix =  make_adjacency_matrix(10,friendpair)
print(adjacency_matrix)

arr = np.array([ 0.089,-0.35 ,  0.515 , 0.151 , 0.144 ,-0.096 , 0.078 , 0.213, -0.693, 0])
arr2 = adjacency_matrix.dot(arr)

arr3 = arr*2.669

print(arr2,arr3)


eigen_values,eigen_vector = np.linalg.eig(adjacency_matrix)
# eigen_values,eigen_vector = np.linalg.eig(np.array(X))

eigen_vectors = np.around(eigen_vector, 3)
eigen_values = np.around(eigen_values, 3)

eigen_vectors = eigen_vector[:,2]

node_size = [ 0 if eigen_vector < 0 else eigen_vector*2000 for eigen_vector in eigen_vectors]

draw_basic_network_graph(friendpair,node_size)
print(eigen_values,eigen_vector)



