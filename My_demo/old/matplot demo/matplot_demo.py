"""
python_visual_animation.py by xianhu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# 解决中文乱码问题
myfont = fm.FontProperties(fname="/Library/Fonts/Songti.ttc", size=14)
matplotlib.rcParams["axes.unicode_minus"] = False


def three_dimension_scatter():
    """
    3d scatter plot
    """
    # 生成画布
    fig = plt.figure()

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(50):
        # 清除原有图像
        fig.clf()

        # 设定标题等
        fig.suptitle("三维动态散点图", fontproperties=myfont)

        # 生成测试数据
        point_count = 100



        x = np.random.random(point_count)
        y = np.random.random(point_count)
        z = np.random.random(point_count)



        color = np.random.random(point_count)
        scale = np.random.random(point_count) * 100

        # 生成画布
        ax = fig.add_subplot(111, projection="3d")

        # 画三维散点图
        ax.scatter(x, y, z, s=scale, c='r', marker="o")

        # 设置坐标轴图标
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        # 设置坐标轴范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return
three_dimension_scatter()