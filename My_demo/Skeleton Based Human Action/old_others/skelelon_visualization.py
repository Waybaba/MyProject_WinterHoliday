"""
function:get skeleton data from date file,and visualize them
step1:use a function to get information from another txt file, save it in a matrix
step2:use a loop to picture the date at an proper interval, visualize both the joints and skeleton
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 解决中文乱码问题
# myfont = fm.FontProperties(fname="/Library/Fonts/Songti.ttc", size=14)
# matplotlib.rcParams["axes.unicode_minus"] = False

"""
返回帧数据组，格式为n帧*3坐标*20关节【由n帧*20关节*（x，y，z）转秩得到，虽然后者更直观，但之后绘图不方便】
输入n为要获取的帧数，文件中即为行数
"""
def get_date_list(n):
    # 打开一个文件
    fo = open("../Skeleton Based Human Action/joints/joints_s04_e02.txt","r+")  # r+是读写模式，比较常用
    print("文件名: ", fo.name)
    print("是否已关闭 : ", fo.closed)
    print("访问模式 : ", fo.mode)

    # fo.write( "www.runoob.com!\nVery good site!\n")
    date_list = np.zeros(shape=(n, 3, 20))  # 创建总长为n的数据列

    for i in range(n):  # 从0开始，不包括括号里的数,读取n行
        # 提取出来一行的整字符串
        str = fo.readline()
        # print("第", end="")
        # print(i + 1, end="")
        # print("行的字符是：", end="")
        # print(str)

        # 分割成61元数组，转化成数字格式
        split_str = str.split()
        for j in range(len(split_str)):
            split_str[j] = float(split_str[j])
        # print(split_str)
        # print(len(split_str))
        xyz_list = []

        # 三元分割
        for j in range(len(split_str)):
            if (j - 1) % 3 == 0:
                xyz_list.append([split_str[j], split_str[j + 1], split_str[j + 2]])
                # ax.scatter(split_str[j],split_str[j+1],split_str[j+2], color="r", marker="^")
        # print(xyz_list)

        # 转置，换成numpy
        xyz_numpy = np.array(xyz_list)
        # print(xyz_numpy)
        xyz_numpy = xyz_numpy.T
        # print(xyz_numpy)

        # 将本行/本帧添加到总的数据列里面
        date_list[i] = xyz_numpy
    print("date_load finish!!")
    # print("总的数据列为：")
    # print(date_list)

    #关闭文件
    fo.close()
    print("是否已关闭 : ", fo.closed)
    return date_list


"""
主程序
"""
#可调参数
frame_sum=500#要纳入的帧数
#调取函数，从文件中获取数据
date_list=get_date_list(frame_sum)
# 绘图初始化
fig = plt.figure()
# 打开交互模式
plt.ion()
#逐帧操作
for frame in range(frame_sum):
    # 准备
    fig.clf()#清空原来的
    fig.suptitle("3D Skeletion ")#标题
        # fig.suptitle("三维动态散点图", fontproperties=myfont)#这样写可以输入中文，但是我没有字体
    point_count = 100
    ax = fig.add_subplot(111, projection="3d")
    #点绘图
    ax.scatter(date_list[frame][2][3], date_list[frame][0][3], date_list[frame][1][3], c="r", s=500, marker="o")#画头部，因为特别，所以画大点
    ax.scatter(date_list[frame][2],date_list[frame][0],date_list[frame][1])#画其他joints
    #点标签
    for point_index in range(20):
        # #配置单点的数据
        x = date_list[frame][2][point_index]
        y = date_list[frame][0][point_index]
        z = date_list[frame][1][point_index]

        # color = np.random.random(point_count)
        # scale = np.random.random(point_count) * 100
        label = str(point_index)
        # # 标上点
        # ax.scatter(x, y, z, s=point_index, c='r', marker="o",label="123")
        # ax.text(x,y,z,label)
    #线绘图
    line_index = [2, 3, 2, 4, 2, 8, 5, 4, 5, 6, 9, 8, 9, 10,10,11,16,17,13,14,17,18,12,13,0,1,1,2,2,12,12,16,16,2,4,12,8,16]
    for line_number in range( len(line_index) ):
        if line_number%2==0 :
            ax.plot(
                (date_list[frame][2][ line_index[line_number] ],date_list[frame][2][ line_index[line_number+1] ]),
                (date_list[frame][0][line_index[line_number]], date_list[frame][0][line_index[line_number + 1]]),
                (date_list[frame][1][line_index[line_number]], date_list[frame][1][line_index[line_number + 1]]),
            )
    # 设置坐标轴标签
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    # 设置坐标轴范围
    ax.set_xlim(2, 5)
    ax.set_ylim(-4, -1)
    ax.set_zlim(0, 2)
    # 暂停
    plt.pause(0.05)

# 关闭交互模式
plt.ioff()
# 贴xyz标签，显示图像
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()#通常做法，但因为是动画，好像交互模式里面就已经涵盖了这种做法
