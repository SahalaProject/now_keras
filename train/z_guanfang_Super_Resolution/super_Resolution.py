# coding:utf-8

import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

"""
使用ESRGAN的图像超分辨率
"""

# Declaring Constants
# IMAGE_PATH = "2.png"


class Super_Resolution():

    def __init__(self, media_path='./'):
        self.super_model = None
        self.media_path = media_path
        self.image_file = ''

    def load_model(self):
        if not self.super_model:
            SAVED_MODEL_PATH = "https://hub.tensorflow.google.cn/captain-pool/esrgan-tf2/1"
            print('load detector model ...')
            self.super_model = hub.load(SAVED_MODEL_PATH)  # 加载模型

    # 定义助手功能
    def preprocess_image(self, image_path):
        """ Loads image from path and preprocesses to make it model ready
            Args:
              image_path: Path to the image file
        """
        hr_image = tf.image.decode_image(tf.io.read_file(image_path))
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[..., :-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

    def save_image(self, image):
        """
          Saves unscaled Tensor Images.
          Args:
            image: 3D image tensor. [height, width, channels]
            filename: Name of the file to save to.
        """
        image_file = str(round(time.time() * 100000000)) + '_Super_Resolution.png'
        image_path = os.path.join(self.media_path, image_file)
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save("%s.jpg" % image_path)
        print("Saved as %s.jpg" % image_path)

    def plot_image(self, image, title=""):
        """
        Plots images from image tensors.
        Args:
          image: 3D image tensor. [height, width, channels].
          title: Title to display in the plot.
        """
        image = np.asarray(image)
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)

    def super_resolution_image(self, IMAGE_PATH):
        # 对从路径加载的图像执行超分辨率
        hr_image = self.preprocess_image(IMAGE_PATH)
        # Plotting Original Resolution image
        # self.plot_image(tf.squeeze(hr_image), title="Original Image")
        # self.save_image(tf.squeeze(hr_image), filename="Original Image")

        start = time.time()
        fake_image = self.super_model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))

        # Plotting Super Resolution Image
        # self.plot_image(tf.squeeze(fake_image), title="Super Resolution")
        self.save_image(tf.squeeze(fake_image))


if __name__ == '__main__':
    IMAGE_PATH = "112.jpg"

    sup_ = Super_Resolution()
    sup_.load_model()
    for i in range(1):
        sup_.super_resolution_image(IMAGE_PATH)
