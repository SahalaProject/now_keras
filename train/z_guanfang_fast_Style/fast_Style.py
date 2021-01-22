import functools
import os
import time

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# print("TF Version: ", tf.__version__)
# print("TF-Hub version: ", hub.__version__)
# print("Eager mode enabled: ", tf.executing_eagerly())
# print("GPU available: ", tf.test.is_gpu_available())

"""
快速转换任意样式的样式
"""


class Fast_Style():
    # @title Define image loading and visualization functions  { display-mode: "form" }

    def __init__(self):
        self.hub_module = None

    def crop_center(self, image):
        """Returns a cropped square image."""
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
        return image

    @functools.lru_cache(maxsize=None)
    def load_image(self, image_url, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads and preprocesses images."""
        # Cache image file locally.
        if image_url.startswith('http'):
            image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
        else:
            image_path = image_url
        # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
        img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
        if img.max() > 1.0:
            img = img / 255.
        if len(img.shape) == 3:
            img = tf.stack([img, img, img], axis=-1)
        img = self.crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img


    def show_n(self, images, image_file, is_show=True, titles=('',)):
        """
        展示和保存图片
        :param images:
        :param image_file:
        :param is_show: 是否展示
        :param titles:
        :return:
        """
        n = len(images)
        image_sizes = [image.shape[1] for image in images]
        w = (image_sizes[0] * 6) // 320
        plt.figure(figsize=(w * n, w))
        gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)

        for i in range(n):
            plt.subplot(gs[i])
            plt.imshow(images[i][0], aspect='equal')
            plt.axis('off')
            plt.title(titles[i] if len(titles) > i else '')
            plt.savefig(image_file)
        plt.show() if is_show else None


    def load_style_hub_model(self):
        hub_handle = 'https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/2'
        if not self.hub_module:
            self.hub_module = hub.load(hub_handle)


    def show_hub_style(self, content_image_url, style_image_url, output_image_size, media_path):
        # 让我们获得一些可玩的图像。
        # @title Load example images  { display-mode: "form" }

        # content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
        # style_image_url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1606911035969&di=18fedb134eabef9792dc279ff302b74f&imgtype=0&src=http%3A%2F%2Fimg1.cfbond.com%2Fgroup1%2FM00%2F00%2F66%2FwKgBlVtX56-APQQqAAFcnBbeSMA884.jpg'  # @param {type:"string"}
        # content_image_url = '11.jpg'  # @param {type:"string"}
        # style_image_url = '137.jpg'  # @param {type:"string"}
        # output_image_size = 384  # @param {type:"integer"}

        # The content image size can be arbitrary.
        content_img_size = (output_image_size, output_image_size)
        # The style prediction model was trained with image size 256 and it's the
        # recommended image size for the style image (though, other sizes work as
        # well but will lead to different results).
        style_img_size = (256, 256)  # Recommended to keep it at 256.

        content_image = self.load_image(content_image_url, content_img_size)
        style_image = self.load_image(style_image_url, style_img_size)
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
        # show_n([content_image, style_image], ['Content image', 'Style image'])

        # 导入TF-Hub模块
        # Load TF-Hub module.

        # 用于图像样式化的该集线器模块的签名为：

        # outputs = hub_module(content_image, style_image)
        # stylized_image = outputs[0]

        # 展示图像风格
        # Stylize content image with given style image.
        # This is pretty fast within a few milliseconds on a GPU.

        outputs = self.hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        # Visualize input images and the generated stylized image.
        image_file = os.path.join(media_path, str(round(time.time() * 100000000)) + '.png')

        # self.show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
        self.show_n([stylized_image], image_file, False, titles=['Stylized image'])  # 只展示合成后图


if __name__ == '__main__':
    # content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
    # style_image_url = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1606911035969&di=18fedb134eabef9792dc279ff302b74f&imgtype=0&src=http%3A%2F%2Fimg1.cfbond.com%2Fgroup1%2FM00%2F00%2F66%2FwKgBlVtX56-APQQqAAFcnBbeSMA884.jpg'  # @param {type:"string"}
    content_image_url = '11.jpg'  # @param {type:"string"}
    style_image_url = 'https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=1651500335,1774686686&fm=26&gp=0.jpg'  # @param {type:"string"}
    output_image_size = 384  # @param {type:"integer"}
    media_path = './'

    fase_style = Fast_Style()
    fase_style.load_style_hub_model()

    for i in range(1):
        fase_style.show_hub_style(content_image_url, style_image_url, output_image_size, media_path)
