# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取图片
# image = plt.imread("D:\\code1\\SoftNetTest\\softnetest\\CASME_sq\\rawpic_crop\s15\\15_0102eatingworms\\img002.jpg")
#
# # 将图片转换为灰度图像
# gray_image = np.mean(image, axis=2)
#
# # 选择小波基函数和分解级别
# wavelet = 'haar'  # 小波基函数，这里选择 Haar 小波
# level = 3  # 分解级别
#
# # 对图像进行离散小波变换
# coeffs = pywt.wavedec2(gray_image, wavelet, level=level)
#
# # 可视化结果
# fig, axes = plt.subplots(level+1, 2, figsize=(10, 10))
# axes[0, 0].imshow(gray_image, cmap='gray')
# axes[0, 0].set_title('Original Image')
# axes[0, 0].axis('off')
#
# cA, (cH, cV, cD) = coeffs[0], coeffs[1]
# axes[1, 0].imshow(cA, cmap='gray')
# axes[1, 0].set_title(f'Approximation 1')
# axes[1, 0].axis('off')
#
# axes[1, 1].imshow(np.abs(cH), cmap='gray')
# axes[1, 1].set_title(f'Horizontal detail 1')
# axes[1, 1].axis('off')
#
# plt.tight_layout()
# plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("dwt2 coefficients", fontsize=14)

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(coeffs2, 'bior1.3')
fig = plt.figure()
plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

# Check that reconstructed image is close to the original
np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)


# Now do the same with dwtn/idwtn, to show the difference in their signatures

coeffsn = pywt.dwtn(original, 'bior1.3')
fig = plt.figure()
for i, key in enumerate(['aa', 'ad', 'da', 'dd']):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(coeffsn[key], interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("dwtn coefficients", fontsize=14)

# Now reconstruct and plot the original image
reconstructed = pywt.idwtn(coeffsn, 'bior1.3')
fig = plt.figure()
plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

# Check that reconstructed image is close to the original
np.testing.assert_allclose(original, reconstructed, atol=1e-13, rtol=1e-13)


plt.show()
