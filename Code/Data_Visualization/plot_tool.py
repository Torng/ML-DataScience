import numpy as np
from matplotlib import pyplot as plt
import matplotlib




def draw_vector(plt_tool:matplotlib.axes,vector:np.array,origin:np.array):
    plt_tool.quiver(*origin, vector[:, 0], vector[:, 1],scale=10)
    # plt_tool.show()



if __name__=="__main__":
    V = np.array([[1, 1]])
    origin = np.array([[0], [0]])  # origin point

    draw_vector(plt,V,origin)