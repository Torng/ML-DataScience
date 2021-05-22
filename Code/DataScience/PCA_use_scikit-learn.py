from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from Code.DataScience.DimensionReduction import de_mean

x = [0,1,2,3,4,5,6,7]
y = [1,1.5,1.7,4,4.5,4.7,8,9]
datas = np.array(list(zip(x,y)))
x_new,y_new = zip(*de_mean(datas))


# df = pd.read_csv("https://bit.ly/2FkIaTv", sep="\t", index_col="名稱")
# lol_data = df.to_numpy()
# lol_data = lol_data[:,1:]

pca = PCA(n_components=1)
pca.fit(de_mean(datas))
X_pca = pca.transform(de_mean(datas))
print(X_pca)
X_new = pca.inverse_transform(X_pca)
plt.scatter(x_new, y_new, alpha=0.3)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.7)
# plt.axis('equal')
plt.xlim(-5,9)
plt.ylim(-5,9)
plt.show()

print(X_new)
print("original shape:   ", datas.shape)
print("transformed shape:", X_pca.shape)