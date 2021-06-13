import tensorflow as tf
from typing import List
from sklearn.model_selection import train_test_split
import numpy as np
def binary_encode(x:int)->List[float]:
    binary:List[float] = []

    for i in range(10):
        binary.append(x%2)
        x = x//2
    return np.array(binary,dtype=float)
def fizz_buzz_encode(x:int):
    if(x%15==0):
        return np.array([0.0,0.0,0.0,1.0])
    elif(x%5==0):
        return np.array ([0.0, 0.0, 1.0, 0.0])
    elif(x%3==0):
        return np.array ([0., 1., 0., 0.])
    else:
        return np.array ([1.,0.,0.,0.])
xs = np.array([binary_encode(n) for n in range(101,1024)])
ys = np.array([fizz_buzz_encode(n) for n in range(101,1024)])



x_train,x_test,y_train,y_test = train_test_split(xs,ys,test_size=0.25)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=256,input_shape=(1,10), activation='relu'),
tf.keras.layers.Dense(units=512,input_shape=(1,10), activation='relu'),
  tf.keras.layers.Dense(units=4,activation="softmax")
])
model.compile(optimizer='adam',
              loss="mean_squared_error",
              metrics=['accuracy'])

model.summary()
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train, epochs=600,verbose=2)
model.evaluate(x_test,y_test)
print(x_test)
print(model.predict([[1,0,0,0,0,0,0,0,0,0]]).round())
