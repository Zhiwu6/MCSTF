import pywt
import numpy as np
import pywt
import cv2

db3 = pywt.Wavelet('db3')  # 创建一个小波对象
print(db3)
"""
Family name:    Daubechies
Short name:     db
Filters length: 6             #滤波器长度
Orthogonal:     True          #正交
Biorthogonal:   True          #双正交
Symmetry:       asymmetric    #对称性，不对称
DWT:            True          #离散小波变换
CWT:            False         #连续小波变换
"""


def dwt_and_idwt():
    '''
    DWT 与 IDWT （离散的小波变换=>分解与重构）
    使用db2 小波函数做dwt
    '''

    x = [3, 7, 1, 1, -2, 5, 4, 6]
    cA, cD = pywt.dwt(x, 'db2')  # 得到近似值和细节系数
    print(cA)  # [5.65685425 7.39923721 0.22414387 3.33677403 7.77817459]
    print(cD)  # [-2.44948974 -1.60368225 -4.44140056 -0.41361256  1.22474487]
    print(pywt.idwt(cA, cD, 'db2'))  # [ 3.  7.  1.  1. -2.  5.  4.  6.]

    # 传入小波对象，设置模式
    w = pywt.Wavelet('sym3')
    cA, cD = pywt.dwt(x, wavelet=w, mode='constant')
    print(cA, cD)
    print(pywt.Modes.modes)
    # [ 4.38354585  3.80302657  7.31813271 -0.58565539  4.09727044  7.81994027]
    # [-1.33068221 -2.78795192 -3.16825651 -0.67715519 -0.09722957 -0.07045258]
    # ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']

    print(pywt.idwt([1, 2, 0, 1], None, 'db3', 'symmetric'))
    print(pywt.idwt([1, 2, 0, 1], [0, 0, 0, 0], 'db3', 'symmetric'))
    # [ 0.83431373 -0.23479575  0.16178801  0.87734409]
    # [ 0.83431373 -0.23479575  0.16178801  0.87734409]


def wavelet_packets():
    # 小波包 wavelet packets
    X = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=X, wavelet='db3', mode='symmetric', maxlevel=3)
    print(wp.data)  # [1 2 3 4 5 6 7 8 9]
    print(wp.level)  # 0    #分解级别为0
    print(wp['ad'].maxlevel)  # 3

    # 访问小波包的子节点
    # 第一层：
    print(wp['a'].data)  # [ 4.52111203  1.54666942  2.57019338  5.3986205   8.20681003 11.18125264]
    print(wp['a'].path)  # a

    # 第2 层
    print(wp['aa'].data)  # [ 3.63890166  6.00349136  2.89780988  6.80941869 15.41549196]
    print(wp['ad'].data)  # [ 1.25531439 -0.60300027  0.36403471  0.59368086 -0.53821027]
    print(wp['aa'].path)  # aa
    print(wp['ad'].path)  # ad

    # 第3 层时：
    print(wp['aaa'].data)
    print([node.path for node in wp.get_level(3, 'natural')])  # 获取特定层数的所有节点,第3层有8个
    # ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

    # 依据频带频率进行划分
    print([node.path for node in wp.get_level(3, 'freq')])
    # ['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']

    # 从小波包中 重建数据
    X = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=X, wavelet='db1', mode='symmetric', maxlevel=3)
    print(wp['ad'].data)  # [-2,-2]

    new_wp = pywt.WaveletPacket(data=None, wavelet='db1', mode='symmetric')
    new_wp['a'] = wp['a']
    new_wp['aa'] = wp['aa'].data
    new_wp['ad'] = wp['ad'].data
    new_wp['d'] = wp['d']
    print(new_wp.reconstruct(update=False))
    # new_wp['a'] = wp['a']  直接使用高低频也可进行重构
    # new_wp['d'] = wp['d']
    print(new_wp)  #: None
    print(new_wp.reconstruct(update=True))  # 更新设置为True时。
    print(new_wp)
    # : [1. 2. 3. 4. 5. 6. 7. 8.]

    # 获取叶子结点
    print([node.path for node in new_wp.get_leaf_nodes(decompose=False)])
    print([node.path for node in new_wp.get_leaf_nodes(decompose=True)])
    # ['aa', 'ad', 'd']
    # ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

    # 从小波包树中移除结点
    dummy = wp.get_level(2)
    for i in wp.get_leaf_nodes(False):
        print(i.path, i.data)
    # aa [ 5. 13.]
    # ad [-2. -2.]
    # da [-1. -1.]
    # dd [-1.11022302e-16  0.00000000e+00]
    node = wp['ad']
    print(node)  # ad: [-2. -2.]
    del wp['ad']  # 删除结点
    for i in wp.get_leaf_nodes(False):
        print(i.path, i.data)
    # aa [ 5. 13.]
    # da [-1. -1.]
    # dd [-1.11022302e-16  0.00000000e+00]

    print(wp.reconstruct())  # 进行重建
    # [2. 3. 2. 3. 6. 7. 6. 7.]

    wp['ad'].data = node.data  # 还原已删除的结点
    print(wp.reconstruct())
    # [1. 2. 3. 4. 5. 6. 7. 8.]

    assert wp.a == wp["a"]
    print(wp["a"])
    # a: [ 2.12132034  4.94974747  7.77817459 10.60660172]


# if __name__ == '__main__':
#     dwt_and_idwt()
#     wavelet_packets()
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


def haar_img():
    img_u8 = cv2.imread("D:\\code1\\SoftNetTest\\softnetest\\CASME_sq\\rawpic_crop\s15\\15_0102eatingworms\\img002.jpg")

    img_f32 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img_f32, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    return img


if __name__ == '__main__':
    img = haar_img()

    plt.imshow(img, 'gray')
    plt.title('img')
    plt.show()
