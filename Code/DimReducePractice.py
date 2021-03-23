import numpy as np
from typing import List
import math
import matplotlib.pyplot as plt
import pandas as pd
from Code.DimensionReduction import de_mean,pca,transform,first_principal_component,project,direction


# df = pd.read_csv("https://bit.ly/2FkIaTv", sep="\t", index_col="名稱")
# lol_data = df.to_numpy()
# lol_data = lol_data[:,1:]
#
#
#
#
# new_data = pca(lol_data,11)
# print(new_data)




x = [0,1,2,3,4,5,6,7]
y = [1,1.5,1.7,4,4.5,4.7,8,9]
datas = list(zip(x,y))
# trans_data = transform(datas,new_data)
# x_new,y_new = zip(*trans_data)

pca = pca(de_mean(datas),1)
print("pca",pca)
new_pca = transform(de_mean(datas),pca)
print(de_mean(datas))
print(new_pca)
x_new,y_new = zip(*de_mean(datas))
plt.scatter(x_new, y_new,alpha=0.3)
components = np.array(pca[0])
p_x_new,p_y_new = zip(*[components*p_len for p_len in new_pca])
plt.xlim(-5,9)
plt.ylim(-5,9)
plt.scatter(p_x_new, p_y_new,alpha=0.7)
plt.show()