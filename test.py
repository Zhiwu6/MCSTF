import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from training import *


def visualize_feature_maps(model, input_image_1, input_image_2, input_image_3):
    # 创建一个模型，输出某一层的特征图
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

    # 输入图像，获取特定层的输出
    activations = activation_model.predict([input_image_1, input_image_2, input_image_3])

    # 可视化特征图
    for layer_activation in activations:
        if len(layer_activation.shape) < 4:
            continue
        n_features = layer_activation.shape[-1]  # 特征图中的特征数量
        size = layer_activation.shape[1]  # 特征图的尺寸
        n_cols = 16 // size
        display_grid = np.zeros((size * n_cols, size * n_features // n_cols))
        for col in range(n_cols):
            for row in range(n_features // n_cols):
                channel_image = layer_activation[0, :, :, col * n_cols + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_activation.shape)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1)  # 假设输入图像是单通道的灰度图像
    image = tf.image.resize(image, (42, 42))  # 将图像调整为模型接受的大小
    image = tf.expand_dims(image, axis=0)  # 添加批次维度
    return image
def tttest():
    inputs1 = layers.Input(shape=(42, 42, 1))
    conv1 = layers.Conv2D(3, (5, 5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv1)
    # channel 2
    inputs2 = layers.Input(shape=(42, 42, 1))
    conv2 = layers.Conv2D(5, (5, 5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv2)
    # channel 3
    inputs3 = layers.Input(shape=(42, 42, 1))
    conv3 = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=merged)


image_path_1 = "D:\\Figure_01"
image_path_2 = "D:\\Figure_03"
image_path_3 = "D:\\Figure_01"



