import numpy as np
from typing import List
import math
import matplotlib.pyplot as plt
import pandas as pd
from Code.DimensionReduction import de_mean,pca,transform,first_principal_component

x = [0,1,2,3,4,5,6,7]
y = [1,1.5,1.7,3,4,5,5.5]
# df = pd.read_csv("https://bit.ly/2FkIaTv", sep="\t", index_col="名稱")
# lol_data = df.to_numpy()
# lol_data = lol_data[:,1:]

datas = list(zip(x,y))
x_new,y_new = zip(*de_mean(datas))
# print(x_new,y_new)

# new_data = pca(datas,2)
# trans_data = transform(datas,new_data)
# print("========")
# print(trans_data)
# x_new,y_new = zip(*trans_data)
plt.scatter(x_new, y_new)
# plt.xticks([-2,-1,0,1,2,3,4,5]) #設定x軸刻度
# plt.yticks([-2,-1,0,1,2,3,4,5])
plt.xlim(-5,9)
plt.ylim(-5,9)
v_1,v_2 = first_principal_component(de_mean(datas),1000)

plt.quiver(0,0,v_1,v_2, color=['r'], scale=2)
plt.show()