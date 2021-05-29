from typing import Tuple,List
import os
import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import ntpath
import time
train_path = "/Users/HawkTorng/Desktop/DataSet/ASL/asl_alphabet_train/asl_alphabet_train"
test_path = "/Users/HawkTorng/Desktop/DataSet/ASL/asl_alphabet_test/asl_alphabet_test"

def get_train_datas(path:str)->Tuple[List[np.array],List[np.array]]:
    dir_list = os.listdir(path)
    x_train = []
    y_train = []
    for dir in dir_list:
        if(dir==".DS_Store"):
            continue
        start = time.time()
        for file in glob.glob(os.path.join(path,dir)+"/*.jpg"):
            image = Image.open(file)
            x_train.append(np.asarray(image)/255.0)
            y_train.append(str(dir))
        end = time.time()
        print(end-start)
    return  [x_train,y_train]
def get_test_datas(path:str)->[List[np.array],List[str]]:
    x_test = []
    y_test = []
    for file in glob.glob(path+"/*.jpg"):
        image = Image.open(file)
        x_test.append(np.asarray(image))
        file_name = ntpath.basename(file)
        label = file_name.split("_")[0]
        y_test.append(label)
    return [x_test,y_test]
def create_layers():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200,200,3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(29,activation="softmax")
    ])
    return model


if __name__ == "__main__":
    x_train,y_train = get_train_datas(train_path)
    x_test,y_test = get_test_datas(test_path)
    model = create_layers()
    tf.keras.utils.plot_model(model,show_shapes=True)
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))
