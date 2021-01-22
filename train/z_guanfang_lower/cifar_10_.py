import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# 卷积神经网络（CNN）

# 下载并准备CIFAR10数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# 验证数据
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.imsave('2.png', train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
# plt.show()


from a_util.ndarray_save_image_by_class import ndarray_save_image_by_class
ndarray_save_image_by_class(10000 , 'testing', class_names, train_labels, train_images)

# 创建卷积基础
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
# # 到目前为止，让我们展示模型的架构。
# model.summary()
#
# model.add(layers.BatchNormalization())
# model.add(layers.AlphaDropout(rate=0.5))
#
# # 在顶部添加密集层
# # 为了完成我们的模型，您将从卷积基数（形状为（4，4，64））的最后一个输出张量馈入一个
# # 或多个Dense层以执行分类。密集层将向量作为输入（一维），而当前输出是3D张量。首先，
# # 将3D输出展平（或展开）为1D，然后在顶部添加一个或多个Dense层。CIFAR具有10个输出类，
# # 因此您使用具有10个输出和softmax激活的最终Dense层。
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
#
# # 这是我们模型的完整架构。
# model.summary()
#
# # 如您所见，我们的（4，4，64）输出在经过两个密集层之前被展平为形状为（1024）的向量。
#
# # 编译和训练模型
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(train_images, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))
#
# # 评估模型
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)

