from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3,shuffle=True)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(4,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# model每層定義好後需要經過compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_y, epochs=40)

model.evaluate(test_X, test_y, verbose=2)