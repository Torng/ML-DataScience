import numpy as np
from PIL import Image
from Code.k_means import KMeans
from matplotlib import pyplot as plt
image = Image.open(r"/Users/HawkTorng/Downloads/海綿寶寶.jpeg")
image_arr = np.array(image)
image_arr = image_arr/256
# print(np.shape(image_arr))
image_flatten = [pixel.tolist() for row in image_arr for pixel in row]
# print(np.shape(image_flatten))
# print(image_flatten)

model = KMeans(5)
model.train(image_flatten)
print(len(model.means))


# def recolor(pixel:np.array):
#     cluster = model.classify(pixel)
#     return model.means[cluster]
# print(recolor(np.array([0.63671875,0.81640625,0.94921875])))
# new_img = [recolor(pixel)for arr in image_arr]
# for arr in image_arr:
#     for pixel in arr:
#         print(pixel)
# print(new_img)
# plt.imshow(new_img)
#
# plt.axis('off')
# plt.show()
