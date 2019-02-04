"""
function:get skeleton data from date file,and visualize them
step1:use a function to get information from another txt file, save it in a matrix
step2:use a loop to picture the date at an proper interval, visualize both the joints and skeleton

socks :
    show_gif(x): show the skeleton gif according to the numpy
        input : 50*25*3c numpy
    get_date_list : distill a file and get the skeleton position date
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import is_for_import.ntu_date_preprocess_2 as ntu

# 解决中文乱码问题
# myfont = fm.FontProperties(fname="/Library/Fonts/Songti.ttc", size=14)
# matplotlib.rcParams["axes.unicode_minus"] = False

"""
返回帧数据组，格式为n帧*3坐标*20关节【由n帧*20关节*（x，y，z）转秩得到，虽然后者更直观，但之后绘图不方便】
输入n为要获取的帧数，文件中即为行数
"""
def get_date_list(n=50):
    #---打开一个文件
    fo = open("/Users/Waybaba/Library/Mobile Documents/com~apple~CloudDocs/科研/Machine Learning/MyProject/Skeleton Based Human Action/ntu_database_part/S001C001P001R001A007.skeleton","r+")  # r+是读写模式，比较常用
    print("文件名: ", fo.name)
    print("是否已关闭 : ", fo.closed)
    print("访问模式 : ", fo.mode)
    # fo.write( "www.runoob.com!\nVery good site!\n")
    date_list = np.zeros(shape=(n, 3, 25))  # 创建总长为n的数据列

    #---逐行读取，提取数据存入
    fo.readline()#skip the first line
    for i in range(n):
        # print("reading the frame:",end="")
        # print(i)
        fo.readline()
        fo.readline()
        fo.readline()

        one_frame_xyz_list = []
        for m in range(25):#read 25 lines, and give back a 25*xzy number
            # print("reading the skeleton:", end="")
            # print(m)
            str_oneline=fo.readline()
            split_str_oneline=str_oneline.split()#split into sigle
            xyz_oneline = [split_str_oneline[0],split_str_oneline[1],split_str_oneline[2]]#get the xyz
            for str_number in range(len(xyz_oneline)):#chage into float
                xyz_oneline[str_number]=float(xyz_oneline[str_number])
            one_frame_xyz_list.append(xyz_oneline)
        xyz_list_numpy = np.array(one_frame_xyz_list)
        xyz_list_numpy = xyz_list_numpy.T
        date_list[i]=xyz_list_numpy
    print("date load finish!!!")
    #关闭文件
    fo.close()
    print("是否已关闭 : ", fo.closed)
    return date_list

"""
主程序
"""
#可调参数

def show_old(date_list):
    frame_sum=50#要纳入的帧数
    #调取函数，从文件中获取数据
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
        # 线绘图
        line_index = [17,18,18,19,13,14,14,15,1,2,2,3,3,4]
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


def show_gif(x):
    date_list = np.zeros(shape=(50, 3, 25))
    for index in range(50):
        date_list[index] = x[index].T
    show_old(date_list)
def test():
    input_train, lable_train, input_test, lable_test = ntu.load_date("4_actions")
    show_gif(input_train[0])

# test()