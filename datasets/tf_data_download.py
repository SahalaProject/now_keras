import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


"""
TensorFlow数据集
"""

# data = tensorflow.data.Dataset('food-101.tar.gz')
# tfds.core.DatasetBuilder # 所有数据集构建器都是的子类

# 查找可用的数据集
data_list = tfds.list_builders()
print(data_list)

# 加载数据集，下载路径 C:\Users\liyubin1\tensorflow_datasets  ，下载完将图片拷贝训练
ds = tfds.load('cars196', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

