from tensorflow import keras

import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from tensorflow.python.layers.base import Layer
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)  # SOFTNet default = 0.0005
from model import *
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from model import *
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Attention

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)  # SOFTNet default = 0.0005
loss_fn = tf.keras.losses.MeanSquaredError()

random.seed(1)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply

random.seed(1)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input,Lambda

import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply



import numpy as np
from matplotlib import pyplot as plt
import pywt
import PIL

img = PIL.Image.open(r"D:\\code1\\SoftNetTest\\softnetest\\5547758_eea9edfd54_n.jpg")
img = np.array(img)
LLY,(LHY,HLY,HHY) = pywt.dwt2(img, 'haar')
plt.subplot(2, 2, 1)
plt.imshow(LLY, cmap="Greys")
plt.subplot(2, 2, 2)
plt.imshow(LHY, cmap="Greys")
plt.subplot(2, 2, 3)
plt.imshow(HLY, cmap="Greys")
plt.subplot(2, 2, 4)
plt.imshow(HHY, cmap="Greys")
plt.show()

# def SOFTNet():
# #     inputs1 = layers.Input(shape=(42, 42, 1))
# #     inputs2 = layers.Input(shape=(42, 42, 1))
# #     inputs3 = layers.Input(shape=(42, 42, 1))
# #     input = layers.Concatenate()([inputs1,inputs2,inputs3])
# #     print(input.shape)
# #     model = SLSwinTransformer(
# #         image_size=42,
# #         patch_size=6,
# #         window_size=7,
# #         embed_dim=96,
# #         depths=(2, 2, 6, 2),
# #         num_heads=(3, 6, 12, 24),
# #         num_classes=1,
# #     )
# #     outputs = model(input)
# #     print(outputs.shape)
# #
# #     # # channel 1
# #     # inputs1 = layers.Input(shape=(42, 42, 1))
# #     # multip1 = model(inputs1)
# #     # print("multip1_shape",multip1.shape)
# #     # # channel 2
# #     # inputs2 = layers.Input(shape=(42, 42, 1))
# #     # multip2 = model(inputs2)
# #     # print("multip2_shape",multip2.shape)
# #
# #
# #     # # channel 3
# #     # inputs3 = layers.Input(shape=(42, 42, 1))
# #     # multip3 = model(inputs3)
# #     # print("multip3_shape",multip3.shape)
# #     #
# #     # outputs = layers.Concatenate()([multip1, multip2, multip3])
# #     # print("merged_shape:",merged.shape)
# #     # interpretation
# #     # merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(outputs)
# #     # flat = layers.Flatten()(merged_pool)
# #     # dense = layers.Dense(400, activation='relu')(flat)
# #     # outputs = layers.Dense(1, activation='linear')(merged_pool)
# #     # Takes input u,v,s
# #     model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# #     # model = keras.models.Model(inputs=input, outputs=outputs)
# #
# #     # compile
# #     sgd = keras.optimizers.SGD(lr=0.0005)
# #     model.compile(loss=loss_fn, optimizer=sgd, metrics=["accuracy", tf.keras.metrics.MeanAbsoluteError()])
# #     # model.compile(optimizer=optimizer,loss=loss_fn,metrics=["accuracy", tf.keras.metrics.MeanAbsoluteError()],)
# #     return model
# #     # # channel 1
# #     # inputs1 = layers.Input(shape=(42, 42, 1))
# #     # # model = SLSwinTransformer(
# #     # #     num_classes=2,
# #     # #     image_size=42,
# #     # #     patch_size=1
# #     # # )
# #     # # multip1 = model(inputs1)
# #     # # print("multip1_shape:",multip1.shape)
# #     # # print("inputs1_shape",inputs1.shape)
# #     # # KerasTensor(type_spec=TensorSpec(shape=(None, 42, 42, 1), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
# #     # conv11 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs1)
# #     # pool11 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv11)
# #     # print("pool11尺寸大小为：", pool11.shape)
# #     # # pool11尺寸大小为： (None, 14, 14, 3)
# #     # # pool12尺寸大小为： (None, 14, 14, 3)
# #     # # pool13尺寸大小为： (None, 14, 14, 5)
# #     #
# #     # conv12 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs1)
# #     # pool12 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv12)
# #     #
# #     # conv13 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs1)
# #     # pool13 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv13)
# #     # multip1 = layers.Concatenate()([pool11, pool12, pool13])
# #     # print("multip1:",multip1.shape)
# #     # multip1 = model(multip1)
# #     # print("multip1:",multip1.shape)
# #     # # (None, 14, 14, 11)
# #     # # channel 2
# #     # inputs2 = layers.Input(shape=(42, 42, 1))
# #     #
# #     # conv21 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs2)
# #     # pool21 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv21)
# #     #
# #     # conv22 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs2)
# #     # pool22 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv22)
# #     # conv23 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs2)
# #     # pool23 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv23)
# #     #
# #     # multip2 = layers.Concatenate()([pool21, pool22, pool23])
# #     # print("multip2:",multip2.shape)
# #     # # multip2 = model(inputs2)
# #     # # print("multip2_shape:",multip2.shape)
# #     # multip2 = model(multip2)
# #     # print("multip2:",multip2.shape)
# #     #
# #     #
# #     # # channel 3
# #     # inputs3 = layers.Input(shape=(42, 42, 1))
# #     #
# #     # conv31 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs3)
# #     # pool31 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv31)
# #     #
# #     # conv32 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs3)
# #     # pool32 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv32)
# #     # conv33 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs3)
# #     # pool33 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv33)
# #     #
# #     # multip3 = layers.Concatenate()([pool31, pool32, pool33])
# #     # # multip3 = model(inputs3)
# #     # # print("multip3_shape:",multip3.shape)
# #     # multip3 = model(multip3)
# #     # print("multip3:",multip3.shape)
# #     #
# #     #
# #     # # print("multip3:",multip3.shape)
# #     # # multip3: (None, 14, 14, 11)
# #     # # merge
# #     # merged = layers.Concatenate()([multip1, multip2, multip3])
# #     # print("merged:",merged.shape)
# #     # # (None, 14, 14, 33)
# #     # print("脱离苦海了")
# #     # # conv = layers.Conv2D(8, (5, 5), padding='same', activation=None)(merged)
# #     # # # interpretation
# #     # # merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
# #     # # flat = layers.Flatten()(merged_pool)
# #     # # flat = layers.Flatten()(merged)
# #     #
# #     # # dense = layers.Dense(400, activation='relu')(flat)
# #     # outputs = layers.Dense(1, activation='linear')(merged)
# #     # # Takes input u,v,s
# #     # model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# #     # # compile
# #     # sgd = keras.optimizers.SGD(lr=0.0005)
# #     # # model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
# #     # model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
# #     # # print(model.summary())
# #     #
# #     # return model
#
#
# # max_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)
# # print("max_pool shape:", max_pool.shape)  # max_pool shape: (None, 7, 7, 33)
# # avg_pool = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2))(merged)
# # print("avg_pool shape:", avg_pool.shape)  # avg_pool shape: (None, 7, 7, 33)
# #
# # max_pool_Conv1D = layers.Conv1D(filters=3, padding='same', kernel_size=3)(max_pool)
# # print("max_pool_Conv1D shape:", max_pool_Conv1D.shape)  # max_pool_Conv1D shape: (None, 7, 7, 3)
# # avg_pool_Conv1D = layers.Conv1D(filters=3, padding='same', kernel_size=3)(avg_pool)
# # print("avg_pool_Conv1D shape:", avg_pool_Conv1D.shape)  # avg_pool_Conv1D shape: (None, 7, 7, 3)
# # max_avg = layers.Concatenate()([max_pool_Conv1D, avg_pool_Conv1D])
# # print("max_avg shape:", max_avg.shape)  # max_avg shape: (None, 7, 7, 6)
# # merged = layers.Concatenate()()([max_avg, merged])
# # print("------------------")
# # print(merged.shape)
# # # merged = Activation('sigmoid')(max_avg)
# # # print("sigmoid_output___shape:", merged.shape)#sigmoid_output___shape: (None, 7, 7, 6)
# # # print("merged_shape:", merged.shape)#merged_shape: (None, 14, 14, 33)
# # # merged = Concatenate()([sigmoid_output, merged])
# # # print("merged:", merged.shape)
# # # merged = Attention()(merged)
# # # merged = inception_module(merged,4)
# # # merged: (None, 14, 14, 12)
# import torch
# from model import *
#
#
# # merged1: (None, 7, 7, 44)
# # merged2: (None, 6, 6, 44)
# # merged3: (None, 6, 6, 44)
# # (None, 14, 14, 44)
# # outpots_shape: (None, 1)
# # merged1 = layers.Conv2D()
# #
# # mode = SwinTransformer()
# # print(22)
# class Fusion(tf.keras.Model):
#     def __init__(self, num_classes=2):
#         super(Fusion, self).__init__()
#
#         self.featuresA = tf.keras.Sequential([
#             # 1
#             tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 2
#             tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             # 3
#             tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#         ])
#
#         self.featuresB = tf.keras.Sequential([
#             # 1
#             tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 2
#             tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             # 3
#             tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#         ])
#
#         self.fusion_feature = tf.keras.Sequential([
#             # 4
#             tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             # 5
#             tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 6
#             tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 7
#             tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             # 8
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 9
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 10
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             # 11
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 12
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             # 13
#             tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#             tf.keras.layers.GlobalAveragePooling2D(),
#         ])
#
#         self.classifier_test = tf.keras.layers.Dense(num_classes)
#
#         self.classifier = tf.keras.Sequential([
#             # 14
#             tf.keras.layers.Dense(4096, activation='relu'),
#             tf.keras.layers.Dropout(0.5),
#             # 15
#             tf.keras.layers.Dense(4096, activation='relu'),
#             tf.keras.layers.Dropout(0.5),
#             # 16
#             tf.keras.layers.Dense(num_classes),
#         ])
#
#     def call(self, x):
#         outA = self.featuresA(x)
#         outB = self.featuresB(x)
#         out = tf.concat([outA, outB], axis=-1)
#         print(333333333333333333333333333)
#         # out_fusion = self.fusion_feature(out_fusion)
#         # print(444444444444444444444444444444)
#         # out_fusion = tf.reshape(out_fusion, (tf.shape(out_fusion)[0], -1))
#         # print(555555555555555555555555555555)
#         # out = self.classifier_test(out_fusion)
#         return out
# # class fusion(nn.Module):
# #     def __init__(self, num_classes=10):
# #         super(fusion, self).__init__()
# #         self.featuresA = nn.Sequential(
# #             # 1
# #             nn.Conv2d(3, 64, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(True),
# #             # 2
# #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             # 3
# #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(128),
# #             nn.ReLU(True),
# #
# #         )
# #         self.featuresB = nn.Sequential(
# #             # 1
# #             nn.Conv2d(3, 64, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(True),
# #             # 2
# #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             # 3
# #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(128),
# #             nn.ReLU(True),
# #         )
# #         self.fusionFeature = nn.Sequential(
# #             # 4
# #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(256),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             # 5
# #             nn.Conv2d(256, 512, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU(True),
# #             # 6
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU(True),
# #             # 7
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             # 8
# #             nn.Conv2d(512, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             # 9
# #             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             # 10
# #             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             # 11
# #             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             # 12
# #             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             # 13
# #             nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU(True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.AvgPool2d(kernel_size=1, stride=1),
# #         )
# #         self.classifier_test = nn.Linear(1024, 10)
# #         self.classifier = nn.Sequential(
# #             # 14
# #             nn.Linear(1024, 4096),
# #             nn.ReLU(True),
# #             nn.Dropout(),
# #             # 15
# #             nn.Linear(4096, 4096),
# #             nn.ReLU(True),
# #             nn.Dropout(),
# #             # 16
# #             nn.Linear(4096, num_classes),
# #         )
# #         # self.classifier = nn.Linear(512, 10)
# #
# #     def forward(self, x):
# #         outA = self.featuresA(x)
# #         outB = self.featuresB(x)
# #         out_fusion = torch.cat((outA, outB), dim=1)
# #
# #         out_fusion = self.fusionFeature(out_fusion)
# #         out_fusion = out_fusion.view(out_fusion.size(0), -1)
# #         out = self.classifier_test(out_fusion)
# #         return out

# from keras.layers import Conv2D, Input, concatenate, Lambda, add, merge
# from keras.models import Model
# import tensorflow as tf
# from keras.utils.vis_utils import plot_model  # 可视化


# 该模型为粗特征提取部分为多尺度提取，改进的八个DenseBlock(每个denseBlock都有多尺度提取预处理)，特征稠密融合，降维，子像素卷积+卷积超分重建，
# 在每次卷积前有一个多尺度卷积
# 子像素卷积 Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
# def SubpixelConv2D(scale=4):
#     def subpixel_shape(input_shape):
#         dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))]
#         output_shape = tuple(dims)
#         return output_shape

#     def subpixel(x):
#         return tf.depth_to_space(x, scale)  # tensorflow中的子像素卷积

#     return Lambda(subpixel, output_shape=subpixel_shape)


# # 计算psnr
# def psnr(y_true, y_pred):
#     psnr_cal = tf.image.psnr(y_true, y_pred, max_val=1.0)
#     return psnr_cal


# # 多尺度卷积,4倍filters,含有1x1卷积
# def multiConv(block_input, activation='relu', filters=16):
#     initializer = 'he_normal'
#     # initializer = 'glorot_normal'
#     ######################
#     model_scale9 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
#     model_scale9 = Conv2D(filters=filters, kernel_size=(9, 9), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(model_scale9)

#     #####################
#     model_scale7 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度

#     model_scale7 = Conv2D(filters=filters, kernel_size=(7, 7), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(model_scale7)

#     ####################
#     model_scale5 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
#     model_scale5 = Conv2D(filters=filters, kernel_size=(5, 5), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(model_scale5)

#     ####################
#     model_scale3 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(block_input)  # 1x1降维到filters个深度
#     model_scale3 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                           kernel_initializer=initializer, )(model_scale3)

#     model_out = concatenate(axis=-1, inputs=[model_scale3, model_scale5, model_scale7, model_scale9])
#     return model_out


# # 输出chanel数量是filters的八倍
# def dense_block(block_input, activation='relu', filters=16):
#     initializer = 'he_normal'
#     # initializer = 'glorot_normal'
#     # 多尺度融合降维
#     model_0 = multiConv(block_input, filters=4)  # 输入16
#     ####DenseNet
#     model_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                      kernel_initializer=initializer, )(model_0)

#     model_2_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer, )(model_1)
#     model_2 = concatenate(axis=-1, inputs=[model_1, model_2_1])

#     model_3_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer, )(model_2)
#     model_3 = concatenate(axis=-1, inputs=[model_2, model_3_1])

#     model_4_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer)(model_3)
#     model_4 = concatenate(axis=-1, inputs=[model_3, model_4_1])

#     model_5_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer)(model_4)
#     model_5 = concatenate(axis=-1, inputs=[model_4, model_5_1])

#     model_6_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer)(model_5)
#     model_6 = concatenate(axis=-1, inputs=[model_5, model_6_1])

#     model_7_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer)(model_6)
#     model_7 = concatenate(axis=-1, inputs=[model_6, model_7_1])

#     model_8_1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=activation,
#                        kernel_initializer=initializer)(model_7)
#     model_8 = concatenate(axis=-1, inputs=[model_7, model_8_1])
#     return model_8


# def DNSR(h=24, w=24, scale=4):
#     activation = 'relu'  # 'tanh'
#     input = Input([h, w, 1])
#     feature = multiConv(input, filters=16)  # 输出64，底层特征
#     dense_block_1 = dense_block(feature, activation=activation)
#     dense_block_2 = dense_block(dense_block_1, activation=activation)
#     dense_block_2 = concatenate(inputs=[dense_block_1, dense_block_2])  ###
#     dense_block_3 = dense_block(dense_block_2, activation=activation)
#     dense_block_3 = concatenate(inputs=[dense_block_2, dense_block_3])  ###
#     dense_block_4 = dense_block(dense_block_3, activation=activation)
#     dense_block_4 = concatenate(inputs=[dense_block_3, dense_block_4])  ###
#     dense_block_5 = dense_block(dense_block_4, activation=activation)
#     dense_block_5 = concatenate(inputs=[dense_block_4, dense_block_5])  ###
#     dense_block_6 = dense_block(dense_block_5, activation=activation)
#     dense_block_6 = concatenate(inputs=[dense_block_5, dense_block_6])  ###
#     dense_block_7 = dense_block(dense_block_6, activation=activation)
#     dense_block_7 = concatenate(inputs=[dense_block_6, dense_block_7])  ###
#     dense_block_8 = dense_block(dense_block_7, activation=activation)
#     dense_block_8 = concatenate(inputs=[dense_block_7, dense_block_8])
#     # ####
#     out = Conv2D(filters=scale * scale, kernel_size=(1, 1), padding='same')(dense_block_8)  # 特征融合
#     out = SubpixelConv2D(scale)(out)  # 上采样
#     out = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(out)  # reconstruction
#     ####
#     DenseNSR = Model(inputs=[input], outputs=[out])
#     return DenseNSR


# if __name__ == "__main__":
#     model, modelName = DNSR(), 'DenseNetSR'
#     model.summary()
#     plot_model(model, to_file="./data/{}.png".format(modelName), show_layer_names=False, show_shapes=True)