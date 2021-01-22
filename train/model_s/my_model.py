# coding:utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

# 打印name和版本
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


import pathlib
data_dir  = './traffic_4/training-中文'
data_dir  = pathlib.Path(data_dir )

#3、############################## 创建数据集######################################
# 为加载程序定义一些参数
batch_size = 64
image_height = 28
image_width = 28

# 拆分训练集
#  找到属于5个类别的3670个文件。 使用2936个文件进行训练。
# 在开发模型时，最好使用验证拆分。我们将使用80％的图像进行训练，并使用20％的图像进行验证。
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)


# 拆分验证集
# 找到属于5个类别的3670个文件。使用734个文件进行验证。
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)

# class_names在这些数据集的属性中找到类名称。
class_names = train_ds.class_names

# 标准化数据
# from tensorflow.keras import layers

# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# 配置数据集以提高性能
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#4、#############################cnn卷积模型训练######################################
model = keras.models.Sequential()

model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))

model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(4, activation='softmax'))


# 编译模型  计算目标函数
# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # optimizer模型的求解方法， metrics指标


history = model.fit(train_ds, epochs=100, validation_data=val_ds)


model.save(filepath='./save_model/traffic')
# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 3)# 坐标轴范围
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图
