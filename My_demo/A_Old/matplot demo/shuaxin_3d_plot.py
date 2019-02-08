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
    #数据
    x_list= [0.807173 ,0.810992 , 0.830081,  0.780888  ,0.710871 , 0.719921 , 0.693947,
         0.660842 , 0.919233 , 0.965422 , 0.951049 , 0.933585 , 0.755948,  0.801393,
         0.839043 , 0.795848,  0.849598  ,0.789295  ,1.123375  ,1.149348]

    y_list= [-0.063675 ,- 0.005344 , 0.286251 , 0.472613 , 0.244838 , 0.061119 - 0.136373,
         - 0.321661 , 0.157613 ,- 0.043953, - 0.250251, - 0.358771 ,- 0.111753 ,- 0.462684
         - 0.795773 ,- 0.830356 ,- 0.150897, - 0.474751 ,- 0.693605, - 0.769005]
    z_list= [2.001335 , 1.994692 , 1.97427   ,1.947406 , 2.102179  ,2.176251,  2.221404,
         2.259011 , 1.901028 , 1.877656  ,1.865846 , 1.86792   ,2.049536,  2.082246,
         2.027483 , 2.099722 , 1.959686  ,2.14623 ,  2.247396  ,2.272757]
    # 生成画布
    fig = plt.figure()

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(10):
        # 清除原有图像
        fig.clf()

        # 设定标题等
        fig.suptitle("三维动态散点图", fontproperties=myfont)

        # 生成测试数据
        point_count = 100

        ax = fig.add_subplot(111, projection="3d")
        for m in range(10):
        # x = np.random.random(point_count)
        # y = np.random.random(point_count)
        # z = np.random.random(point_count)
            x=x_list[m]
            y=y_list[m]
            z=z_list[m]


            color = np.random.random(point_count)
            scale = np.random.random(point_count) * 100

            # 生成画布


            # 画三维散点图
            ax.scatter(x, y, z, s=2, c='r', marker="o")

            # 设置坐标轴图标`
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        # 设置坐标轴范围
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_zlim(0, 5)

        # 暂停
        # plt.pause(1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return
three_dimension_scatter()