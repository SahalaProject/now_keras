# coding:utf-8

import os
import matplotlib.pyplot as plt


"""
数据集中数组保存成图片

# 下载并准备CIFAR10数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# 验证数据
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
 
ndarray_save_image_by_class(training_file, class_names, train_labels, train_images)               
"""

def ndarray_save_image_by_class(num, training_file, class_names, train_labels, train_images):
    """
    数据集中数组保存成图片
    :param num:  数据数量/循环次数
    :param training_file: 路径 str
    :return:
    """
    # 分类保存
    for i in range(num):

        # 训练/验证/测试 路径
        training_file if os.path.exists(training_file) else os.mkdir(training_file)
        # 获取classname
        class_name = class_names[train_labels[i][0]]
        print(class_name)
        # class 路径
        class_path = os.path.join(training_file, class_name)
        class_path if os.path.exists(class_path) else os.mkdir(class_path)
        # 分类后图片路径
        image_file = os.path.join(class_path, '{}.png'.format(str(i) + '_' + class_name))
        # 保存图片
        plt.imsave(image_file, train_images[i])