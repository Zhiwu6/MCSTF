from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from soft_net import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from collections import Counter
import random


def pseudo_labeling(final_images, final_samples, k):
    pseudo_y = []
    video_count = 0

    for subject in final_samples:
        for video in subject:
            samples_arr = []
            if (len(video) == 0):
                pseudo_y.append([0 for i in range(len(final_images[video_count]) - k)])  # Last k frames are ignored
            else:
                pseudo_y_each = [0] * (len(final_images[video_count]) - k)
                for ME in video:
                    samples_arr.append(np.arange(ME[0] + 1, ME[1] + 1))
                for ground_truth_arr in samples_arr:
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index + k)
                        # Equivalent to if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr)) / len(
                                np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1
                pseudo_y.append(pseudo_y_each)
            video_count += 1

    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print('Total frames:', len(pseudo_y))

    return pseudo_y


def loso(dataset, pseudo_y, final_images, final_samples, k):
    # To split the dataset by subjects
    y = np.array(pseudo_y)
    videos_len = []
    groupsLabel = y.copy()
    prevIndex = 0
    countVideos = 0

    # Get total frames of each video
    for video_index in range(len(final_images)):
        videos_len.append(final_images[video_index].shape[0] - k)

    print('Frame Index for each subject:-')
    for video_index in range(len(final_samples)):
        countVideos += len(final_samples[video_index])
        index = sum(videos_len[:countVideos])
        groupsLabel[prevIndex:index] = video_index
        print('Subject', video_index, ':', prevIndex, '->', index)
        prevIndex = index

    X = [frame for video in dataset for frame in video]
    print('\nTotal X:', len(X), ', Total y:', len(y))
    print("groupsLabel:")
    print(groupsLabel)
    return X, y, groupsLabel


def normalize(images):
    for index in range(len(images)):
        for channel in range(3):
            images[index][:, :, channel] = cv2.normalize(images[index][:, :, channel], None, alpha=0, beta=1,
                                                         norm_type=cv2.NORM_MINMAX)
    return images


def generator(X, y, batch_size=12, epochs=1):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            num_images = end - start
            X[start:end] = normalize(X[start:end])
            u = np.array(X[start:end])[:, :, :, 0].reshape(num_images, 42, 42, 1)
            v = np.array(X[start:end])[:, :, :, 1].reshape(num_images, 42, 42, 1)
            os = np.array(X[start:end])[:, :, :, 2].reshape(num_images, 42, 42, 1)

            yield [u, v, os], np.array(y[start:end])


def shuffling(X, y):
    shuf = list(zip(X, y))
    random.shuffle(shuf)
    X, y = zip(*shuf)
    return list(X), list(y)


def data_augmentation(X, y):
    transformations = {
        0: lambda image: np.fliplr(image),
        1: lambda image: cv2.GaussianBlur(image, (7, 7), 0),
        2: lambda image: random_noise(image),
    }
    y1 = y.copy()
    for index, label in enumerate(y1):
        if (label == 1):  # Only augment on expression samples (label=1)
            for augment_type in range(3):
                img_transformed = transformations[augment_type](X[index]).reshape(42, 42, 3)
                X.append(np.array(img_transformed))
                y.append(1)
    return X, y


#
# def SOFTNet():
#     # channel 1
#     inputs1 = layers.Input(shape=(42, 42, 1))
#
#     conv11 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs1)
#     # print("conv11.shape:",conv11.shape)#conv11.shape: (None, 42, 42, 3)
#     pool11 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv11)
#     # print("pool11.shape:",pool11.shape)
#     # pool11.shape: (None, 14, 14, 3)
#     conv12 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs1)
#     # print("conv12.shape:",conv12.shape)
#     # conv12.shape: (None, 42, 42, 3)
#
#     pool12 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv12)
#     # print("pool12.shape:",pool12.shape)
#     # pool12.shape: (None, 14, 14, 3)
#
#     conv13 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs1)
#     # print("conv13.shape:",conv13.shape)
#     # conv13.shape: (None, 42, 42, 5)
#     pool13 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv13)
#     # print("pool13.shape:",pool13.shape)
#     # pool13.shape: (None, 14, 14, 5)
#
#     multip1 = layers.Concatenate()([pool11, pool12, pool13])
#     print("multip1.shape:", multip1.shape)
#     # multip1.shape: (None, 14, 14, 11)
#
#     # channel 2
#     inputs2 = layers.Input(shape=(42, 42, 1))
#
#     conv21 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs2)
#     pool21 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv21)
#
#     conv22 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs2)
#     pool22 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv22)
#     conv23 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs2)
#     pool23 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv23)
#
#     multip2 = layers.Concatenate()([pool21, pool22, pool23])
#     print("multip2:", multip2.shape)
#     # multip2: (None, 14, 14, 11)
#
#     # channel 3
#     inputs3 = layers.Input(shape=(42, 42, 1))
#
#     conv31 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs3)
#     pool31 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv31)
#
#     conv32 = layers.Conv1D(filters=3, padding='same', kernel_size=42)(inputs3)
#     pool32 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv32)
#     conv33 = layers.Conv1D(filters=5, padding='same', kernel_size=42)(inputs3)
#     pool33 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv33)
#
#     multip3 = layers.Concatenate()([pool31, pool32, pool33])
#     print("multip3:", multip3.shape)
#     # multip3: (None, 14, 14, 11)
#
#     # ------------------------------------------------------------------------
#     multip1_2 = layers.Concatenate()([multip1, multip2])
#     # print("multip1_2", multip1_2.shape)  # multip1_2 (None, 14, 14, 22)
#
#     pool1_2 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(multip1_2)
#     # print("pool1_2", pool1_2.shape)  # pool1_2 (None, 14, 14, 22)
#     conv1_2 = layers.Conv2D(11, 1)(pool1_2)  # conv1_2 (None, 14, 14, 11)
#     conv1_2 = layers.BatchNormalization(axis=-1)(conv1_2)
#     conv1_2 = layers.ReLU()(conv1_2)  # (None, 14, 14, 11)
#
#     conv1_22 = layers.Conv2D(11, 1)(multip1_2)
#     conv1_22 = layers.BatchNormalization(axis=-1)(conv1_22)
#
#     # print("**************conv1_22", conv1_22.shape)  # conv1_22 (None, 14, 14, 11)
#
#     # multip12 = layers.Concatenate()([conv1_22, conv1_2, multip1, multip2])
#     multip12 = layers.Concatenate()([conv1_22, conv1_2])
#     # print("multipl12.shape:", multip12.shape)
#     wei_1 = tf.keras.layers.Activation('sigmoid')(multip12)
#     # print("wei_1", wei_1.shape)  # wei_1 (None, 14, 14, 22)
#     wei_1 = layers.Conv2D(11, 1)(wei_1)
#     # print("---",wei_1.shape)
#     multip11 = tf.multiply(wei_1, multip1)
#     # print("multipl1.shape:", multip1.shape)  # multip1.shape: (None, 14, 14, 33)
#
#     multip12 = tf.multiply(wei_1, multip2)
#     # print("multip2.shape:", multip2.shape)  # multip2.shape: (None, 14, 14, 33)
#
#     T1 = layers.Concatenate()([multip11, multip12])
#     # print("multip12.shape:", multip12.shape)  # multipl2.shape: (None, 14, 14, 66)
#     # -----------------------------------------------------------------------------------------------
#     multip1_3 = layers.Concatenate()([multip1, multip3])
#     # print("multip1_2", multip1_2.shape)  # multip1_2 (None, 14, 14, 22)
#
#     pool1_3 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(multip1_3)
#     # print("pool1_2", pool1_2.shape)  # pool1_2 (None, 14, 14, 22)
#     conv1_3 = layers.Conv2D(11, 1)(pool1_3)  # conv1_2 (None, 14, 14, 11)
#     conv1_3 = layers.BatchNormalization(axis=-1)(conv1_3)
#     conv1_3 = layers.ReLU()(conv1_3)  # (None, 14, 14, 11)
#
#     conv1_13 = layers.Conv2D(11, 1)(multip1_3)
#     conv1_13 = layers.BatchNormalization(axis=-1)(conv1_13)
#     multip13 = layers.Concatenate()([conv1_13, conv1_3])
#     # print("multipl12.shape:", multip12.shape)
#     wei_2 = tf.keras.layers.Activation('sigmoid')(multip13)
#     # print("wei_1", wei_1.shape)  # wei_1 (None, 14, 14, 22)
#     wei_2 = layers.Conv2D(11, 1)(wei_2)
#     # print("---",wei_1.shape)
#     multip21 = tf.multiply(wei_2, multip1)
#     # print("multipl1.shape:", multip1.shape)  # multip1.shape: (None, 14, 14, 33)
#
#     multip22 = tf.multiply(wei_2, multip3)
#     # print("multip2.shape:", multip2.shape)  # multip2.shape: (None, 14, 14, 33)
#
#     T2 = layers.Concatenate()([multip21, multip22])
#     # print("multip12.shape:", multip12.shape)  # multipl2.shape: (None, 14, 14, 66)
#
#     # merge
#     merged = layers.Concatenate()([T1, T2])
#
#     # plt.imshow(merged[0, :, :, 0], cmap='gray')
#     # plt.colorbar()  # 添加颜色条
#     # plt.axis('off')  # 关闭坐标轴
#     # plt.show()
#     # print("-----****merged-----,", merged.shape)#merged, (None, 14, 14, 132)
#
#     # interpretation
#     merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)
#     # print("merged_pool.shape:", merged_pool.shape)  # merged_pool.shape: (None, 7, 7, 8)
#     flat = layers.Flatten()(merged_pool)
#     # print("flat_shape:", flat.shape)  # flat_shape: (None, 392)
#
#     dense = layers.Dense(400, activation='relu')(flat)
#     outputs = layers.Dense(1, activation='linear')(dense)
#     print("outpots_shape:", outputs.shape)  # outpots_shape: (None, 1)
#     # Takes input u,v,s
#     model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
#     # compile
#     sgd = keras.optimizers.SGD(lr=0.0005)
#     # model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
#     model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
#     # print(model.summary())
#
#     return model

# def SOFTNet():
#     inputs1 = layers.Input(shape=(42, 42, 1))
#     conv1 = layers.Conv2D(3, (5, 5), padding='same', activation='relu')(inputs1)
#     pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv1)
#     # channel 2
#     inputs2 = layers.Input(shape=(42, 42, 1))
#     conv2 = layers.Conv2D(5, (5, 5), padding='same', activation='relu')(inputs2)
#     pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv2)
#     # channel 3
#     inputs3 = layers.Input(shape=(42, 42, 1))
#     conv3 = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(inputs3)
#     pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv3)
#     # merge
#     merged = layers.Concatenate()([pool1, pool2, pool3])
#     # merged_array = merged.numpy()
#     # plt.imshow(merged_array, cmap='jet')  # 根据需要选择颜色映射
#     # plt.colorbar()  # 添加颜色条
#     # plt.axis('off')  # 关闭坐标轴
#     # plt.show()
#     # interpretation
#     merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)
#     flat = layers.Flatten()(merged_pool)
#     dense = layers.Dense(400, activation='relu')(flat)
#     outputs = layers.Dense(1, activation='linear')(dense)
#     # Takes input u,v,s
#     model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
#     # compile
#     sgd = keras.optimizers.SGD(lr=0.0005)
#     model.compile(loss="mse", optimizer=sgd, metrics=[tf.keras.metrics.MeanAbsoluteError()])
#     return model


def spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p, show_plot):
    prev = 0
    for videoIndex, video in enumerate(final_samples[subject_count - 1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count - 1] for video in subject])
        print('Video:', countVideo + videoIndex)
        score_plot = np.array(
            result[prev:prev + len(dataset[countVideo + videoIndex])])  # Get related frames to each video
        score_plot_agg = score_plot.copy()

        # Score aggregation
        for x in range(len(score_plot[k:-k])):
            score_plot_agg[x + k] = score_plot[x:x + 2 * k].mean()
        score_plot_agg = score_plot_agg[k:-k]

        # Plot the result to see the peaks
        # Note for some video the ground truth samples is below frame index 0 due to the effect of aggregation, but no impact to the evaluation
        if (show_plot):
            plt.figure(figsize=(15, 4))
            plt.plot(score_plot_agg)
            plt.xlabel('Frame')
            plt.ylabel('Score')
        threshold = score_plot_agg.mean() + p * (
                max(score_plot_agg) - score_plot_agg.mean())  # Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg[:, 0], height=threshold[0], distance=k)
        if (
                len(peaks) == 0):  # Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
            preds.append([0, 0, 0, 0, 0, 0])
        for peak in peaks:
            preds.append([peak - k, 0, peak + k, 0, 0, 0])  # Extend left and right side of peak by k frames
        for samples in video:
            gt.append([samples[0] - k, 0, samples[1] - k, 0, 0, 0, 0])
            total_gt += 1
            if (show_plot):
                plt.axvline(x=samples[0] - k, color='r')
                plt.axvline(x=samples[1] - k + 1, color='r')
                plt.axhline(y=threshold, color='g')
        if (show_plot):
            plt.show()
        prev += len(dataset[countVideo + videoIndex])
        metric_fn.add(np.array(preds), np.array(gt))  # IoU = 0.5 according to MEGC2020 metrics
    return preds, gt, total_gt


def evaluation(preds, gt, total_gt, metric_fn):  # Get TP, FP, FN for final evaluation

    TP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['tp']))
    FP = int(sum(metric_fn.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = total_gt - TP
    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    return TP, FP, FN


def visualize_pool1(pool1_output):
    # 可视化第一个样本的 pool1 输出
    # plt.imshow(pool1_output[0, :, :, 0], cmap='gray')  # 假设只有一个样本并且只显示第一个通道的特征图
    pool1_output_img = pool1_output[0]  # 仅取第一个样本的输出进行可视化，可以根据需要进行调整
    # 显示图像
    plt.imshow(pool1_output_img, cmap='gray')  # 根据需要选择颜色映射
    plt.colorbar()  # 添加颜色条
    plt.axis('off')  # 关闭坐标轴
    plt.show()


def training(X, y, groupsLabel, dataset_name, expression_type, final_samples, k, dataset, train, show_plot):
    print("expression_type:")
    print(expression_type)
    logo = LeaveOneGroupOut()

    logo.get_n_splits(X, y, groupsLabel)

    subject_count = 0
    epochs = 10
    batch_size = 12
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)

    p = 0.55  # From our analysis, 0.55 achieved the highest F1-Score
    model = SOFTNet()

    weight_reset = model.get_weights()  # Initial weights
    for train_index, test_index in logo.split(X, y, groupsLabel):  # Leave One Subject Out

        subject_count += 1
        print('Subject : ' + str(subject_count))

        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]  # Get training set
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]  # Get testing set

        print('------Initializing SOFTNet-------')  # To reset the model at every LOSO testing

        path = 'SOFTNet_Weights\\' + dataset_name + '\\' + expression_type + '\\s' + str(subject_count) + '.hdf5'
        print("path:")
        print(path)
        if (train):
            print("*****************************************")
            # Downsampling non expression samples the dataset by 1/2 to reduce dataset bias
            print('Dataset Labels', Counter(y_train))
            unique, uni_count = np.unique(y_train, return_counts=True)
            rem_count = int(uni_count.max() * 1 / 2)

            # Randomly remove non expression samples (With label 0) from dataset
            rem_index = random.sample([index for index, i in enumerate(y_train) if i == 0], rem_count)
            rem_index += (index for index, i in enumerate(y_train) if i > 0)
            rem_index.sort()
            X_train = [X_train[i] for i in rem_index]
            y_train = [y_train[i] for i in rem_index]
            print('After Downsampling Dataset Labels', Counter(y_train))

            # Data augmentation to the micro-expression samples only
            if (expression_type == 'micro-expression'):
                X_train, y_train = data_augmentation(X_train, y_train)
                print('After Augmentation Dataset Labels', Counter(y_train))

            # Shuffle the training set
            X_train, y_train = shuffling(X_train, y_train)
            optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005)
            loss_fn = nn.MSELoss()

            model.set_weights(
                weight_reset)  # Reset weights to ensure the model does not have info about current subject
            model.fit(
                generator(X_train, y_train, batch_size, epochs),
                steps_per_epoch=len(X_train) / batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=generator(X_test, y_test, batch_size),
                validation_steps=len(X_test) / batch_size,
                shuffle=True,
            )
            # model.save_weights(path)
        else:
            model.load_weights(path)  # Load Pretrained Weights

        result = model.predict(
            generator(X_test, y_test, batch_size),
            steps=len(X_test) / batch_size,
            verbose=1
        )

        # import matplotlib.pyplot as plt
        # 创建一个新的模型，该模型接受与原始模型相同的输入，并输出 merged 层的输出
        # visualization_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('merged').output)

        # # 使用模型进行预测
        # merged_output = visualization_model.predict(generator(X_test, y_test, batch_size),
        #     steps=len(X_test) / batch_size,
        #     verbose=1)  # 替换成您的测试样本数据

        # 将 merged_output 转换为 numpy 数组
        # merged_array = merged_output.numpy()

        # 可视化
        # plt.imshow(merged_array, cmap='jet')  # 使用 'jet' 颜色映射，可以根据需要选择
        # plt.colorbar()  # 添加颜色条
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

        # for epoch in range(epochs):
        #     # 在每个训练周期结束后进行可视化
        #     pool1_output = model.get_layer('merged')(X_train)  # 假设池化层的名称为 'max_pooling2d'，根据实际情况替换
        #     visualize_pool1(pool1_output)

        preds, gt, total_gt = spotting(result, total_gt, final_samples, subject_count, dataset, k, metric_fn, p,
                                       show_plot)
        print("-----------------------")
        print("preds", preds)
        print("gt", gt)
        print("total_gt", total_gt)
        TP, FP, FN = evaluation(preds, gt, total_gt, metric_fn)

        print('Done Subject', subject_count)
    return TP, FP, FN, metric_fn

def final_evaluation(TP, FP, FN, metric_fn):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)

    print('TP:', TP, 'FP:', FP, 'FN:', FN)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:",
          round(metric_fn.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))

# Result if Pre-trained weights are used, slightly different to the research paper

# Final Result for CASME_sq micro-expression
# TP: 18 FP: 327 FN: 39
# Precision =  0.0522
# Recall =  0.3158
# F1-Score =  0.0896
# COCO AP@[.5:.95]: 0.0069

# Final Result for CASME_sq macro-expression
# TP: 91 FP: 348 FN: 209
# Precision =  0.2073
# Recall =  0.3033
# F1-Score =  0.2463
# COCO AP@[.5:.95]: 0.0175

# Final Result for SAMMLV micro-expression
# TP: 41 FP: 323 FN: 118
# Precision =  0.1126
# Recall =  0.2579
# F1-Score =  0.1568
# COCO AP@[.5:.95]: 0.0092

# Final Result for SAMMLV macro-expression
# TP: 60 FP: 231 FN: 273
# Precision =  0.2062
# Recall =  0.1802
# F1-Score =  0.1923
# COCO AP@[.5:.95]: 0.0103
