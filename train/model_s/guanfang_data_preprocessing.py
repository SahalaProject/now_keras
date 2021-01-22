# coding:utf-8

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

"""
加载数据
数据集创建
https://tensorflow.google.cn/tutorials/load_data/images
本教程介绍了如何以三种方式加载和预处理图像数据集。首先，
您将使用高级Keras预处理实用程序和图层来读取磁盘上的映像目录。接下来，
您将使用tf.data从头开始编写自己的输入管道。最后，
您将从TensorFlow Datasets中可用的大型目录中下载数据集。
"""

print(tf.__version__)

#1、############################## 下载鲜花数据集######################################
# 本教程使用了数千张花朵照片的数据集。Flowers数据集包含5个子目录，每个类一个：
import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir  = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)

#2、############################## 检索图片######################################
data_dir  = './cat_dog_2/training'
# data_dir  = './flowers_2/training'
data_dir  = pathlib.Path(data_dir )


# 查看图片数量
data_root  = list(data_dir.glob('*/*.jpg')) # 路径
print(data_root ) # 路径总
print(len(data_root )) # 总数

# 每个目录都包含该花类型的图像。这是一些玫瑰：
roses = list(data_dir.glob('cat/*'))
print(str(roses[0]))

# 查看图片
# img = PIL.Image.open((str(roses[0])))
# img.show()


#3、############################## 创建数据集######################################
# 为加载程序定义一些参数
batch_size = 32
image_height = 128
image_width = 128
channels = 3  # 通道
num_classes = 2

# 拆分训练集
#  找到属于5个类别的3670个文件。 使用2936个文件进行训练。
# 在开发模型时，最好使用验证拆分。我们将使用80％的图像进行训练，并使用20％的图像进行验证。
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset='training',
    seed=7,
    image_size=(image_height, image_width),
    batch_size=batch_size)


# 拆分验证集
# 找到属于5个类别的3670个文件。使用734个文件进行验证。
val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset='validation',
    seed=7,
    image_size=(image_height, image_width),
    batch_size=batch_size)


# class_names在这些数据集的属性中找到类名称。
class_names = train_ds.class_names
print(class_names)


# 可视化数据
# 这是训练数据集中的前9张图像。查看清晰度
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# 标准化数据
from tensorflow.keras import layers

# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# 配置数据集以提高性能
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

##################################数据预处理层 Keras预处理层实现数据增强########################################## horizontal
data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomFlip(mode='horizontal', seed=7, input_shape=(image_height, image_width, channels)), # 随机水平和垂直翻转每个图像
                                      layers.experimental.preprocessing.RandomRotation(0.3, seed=7), # 随机旋转
                                      layers.experimental.preprocessing.RandomZoom(0.3, seed=7), # 随机缩放
                                      layers.experimental.preprocessing.RandomCrop(image_height, image_width, seed=7), # 随机剪切
                                      layers.experimental.preprocessing.CenterCrop(image_height, image_width), # 将图像的中央部分裁剪为目标高度和宽度
                                      # layers.experimental.preprocessing.Normalization(axis=-1), # 按功能进行数据归一化
                                      ])
# 让我们通过将数据增强多次应用于同一幅图像来形象化一些增强示例的外观：
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i].numpy().astype("uint8"))
    plt.axis("off")

#4、#############################模型训练######################################

# model = keras.Sequential([
#     data_augmentation, # 使用增强后数据
#     layers.experimental.preprocessing.Rescaling(1./255),
#     layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'), # 卷积层
#     layers.MaxPooling2D(), # 池化层
#     layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'),
#     layers.MaxPooling2D(),
#     layers.BatchNormalization(),
#     layers.AlphaDropout(rate=0.5),
#     layers.Flatten(),
#     layers.Dense(units=128, activation='selu'),
#     layers.Dense(num_classes)
# ])
model = keras.models.Sequential()
model.add(data_augmentation)
model.add(layers.experimental.preprocessing.Rescaling(1./255))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
# model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
# model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
# model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='selu'))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(num_classes))

# 编译模型  计算目标函数
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.SGD(0.01), metrics=['accuracy']) # optimizer模型的求解方法， metrics指标

# model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


model.save(filepath='./save_model/cat_dog_2_cnn')
# model.save(filepath='./save_model/flowers_2')
# 通过一张图打印出 训练值的变化过程
def polt_learning_curves(history):
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5)) # DataFrame是pd中重要的数据结构, 图大小8和5
    plt.grid(True) # 显示网格
    plt.gca().set_ylim(0, 1)# 坐标轴范围
    plt.show()

polt_learning_curves(history) # 打印值训练值变化图


