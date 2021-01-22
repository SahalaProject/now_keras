# coding:utf-8

import numpy as np
# import os
# import PIL
# import PIL.Image
# import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

"""
加载训练好的模型，识别本地图片--输出类别和相似度值
"""

model = keras.models.load_model('./save_model/life_8_224_200_99.78%')  # 加载训练好的完整模型


# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg" # 网络图片
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)


def get_img_pathlist(data_dir='./life_8/validation', image_name='roses'):
    """
    批量获取图片
    :param data_dir:
    :param image_name: data_dir 的下一级目录
    :return:
    """
    import pathlib
    data_dir = pathlib.Path(data_dir)
    image_path = list(data_dir.glob(image_name + '/*'))
    return image_path


def test_model(image_path, class_names, img_height, img_width):
    """测试你的模型"""
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))  # 将图片加载为PIL格式
    input_array = keras.preprocessing.image.img_to_array(img)  # 将PIL映像实例转换为Numpy数组
    input_array = np.array([input_array])  # 来自load_img中描述
    # input_array = tf.expand_dims(input_array, 0) # Create a batch # 使用expand_dims来将维度加1
    # print('input_array: ',input_array)

    input_array = preprocess_input(input_array)
    predictions = model.predict(input_array)[0]  # 输入测试数据,输出预测结果

    class_index = int(np.argmax(predictions))  # 返回识别后最大值索引
    max_value = predictions[class_index]  # 获取最大index的值, 下面防止分数大于1是*100
    class_score = 100 * np.max(predictions) if max_value <= 1 else np.max(predictions)  # 相似度 返回数组的最大值或沿轴的最大值。
    print("这个图像最有可能是： {} 置信度： {:.2f} %".format(class_names[class_index], class_score))
    return class_names[class_index]


def main_test_image(image_name, data_dir):
    """
    识别主程序
    :param image_name:
    :param data_dir: 文件夹路径 或 精确的图片路径列表
    :return:
    """
    if isinstance(data_dir, list):
        # 单张识别
        image_path_list = data_dir  # 磁盘图片路径
    else:
        # 批量识别
        image_path_list = get_img_pathlist(data_dir=data_dir, image_name=image_name)

    error_img_list = []
    for image_path in image_path_list:
        if '.DS_Store' not in str(image_path):
            error_img = test_model(image_path, class_names, img_height, img_width)
            if image_name != error_img:
                print('识别错误的图片：', image_path)
                error_img_list.append(image_path)  # 统计失败个数
    print('error_img_list: ', len(error_img_list))
    print('image_path_list: ', len(image_path_list))
    if len(error_img_list) > 0:
        test_res = '{0} 批量识别准确率：{1} %'.format(image_name, 1 - len(error_img_list) / len(image_path_list))
    else:
        test_res = '{0} 批量识别准确率：{1} %'.format(image_name, 100 // 100)
    print(test_res)
    return test_res


if __name__ == '__main__':
    # 配置 输入尺寸大，特征多，识别率高
    img_height = 224
    img_width = 224
    class_names = ['air', 'bike', 'boat', 'bus', 'car', 'cat', 'dog', 'train']  # 训练时类型 顺序必须和文件夹顺序一致

    # 识别
    # data_dir = ['./img/302_20201103_1107_24801384.jpg']
    # data_dir = get_img_pathlist(data_dir='./life_8/validation', image_name='bike') # 通上一次获取所有验证图
    data_dir = './life_8/validation'  # training
    all_list = []
    for image_name in class_names:
        test_res = main_test_image(image_name, data_dir)
        all_list.append(test_res)

    print('多分类识别统计：', all_list)
