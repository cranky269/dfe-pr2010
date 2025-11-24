import numpy as np
import matplotlib.pyplot as plt


def show_cool_warm_photo(data, cmap='coolwarm',title=None,threshold = None,if_show=False):
    '''
    : param data: 图片数据
    : param cmap: 颜色映射
    : param title: 图片标题
    : param threshold: 显示的数据范围
    : param if_show: 是否显示图片
    : return: 无返回值
    '''
    if threshold is not None:
        data = np.where((data <= threshold[1]) & (data >= threshold[0]), data,0)  # 去除超过threshold的
    im = plt.imshow(data, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # 隐藏坐标轴
    plt.colorbar(im,fraction=0.046, pad=0.04)  # 添加颜色条
    if if_show:
        plt.show()