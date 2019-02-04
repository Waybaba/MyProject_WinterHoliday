# encoding: utf-8
"""
function:change the date form and make it the input of network
step1:
step2:
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import random
import pickle

"""
先找一下文件夹的结构，提取两个动作的所有数据
写一个由整个文件，结合文件名获取到所有帧数据包括lable等信息的函数，长度可以之后处理
取6throw 7pickup吧
直接定位到原来的文件夹下面
文件筛选：根据名字，打开所有文件，遍历文件名，找到对于动作的，将文件名记录在列表里面，之后拼接调用
创建实例，不断更换路径，返回所有帧，并且进行整流成50格,然后append到总数据里面，先按比例向下取整
bug先要删掉错误的数据，好像有些数据有错,导致读取错误；已经完成
bug有些相同动作是双人的，遇到这种数据要先跳过， 完成
bug还有些错误，好像不多，直接跳过把，return error 完成
bug最后的数据shape只能得出第一维的，原因是帧没有完全用numpy数组处理，所以以后要全部用numpy写 完成,md好像是我没有打印traindate的shape而是lable的shape
目前已经可以完美的把一个动作所有的数据提取出 数据数n*50帧*25关节*3坐标的数据库，之后可以再整合一下。

"""


# 输入n*3*25的数据，返回50*3*25的数据
def frame_cut(input_frames):
    frames_count = input_frames.shape[0]  # 这是
    frames_left_index = np.zeros(shape=50)
    output_frames = np.zeros(shape=(50, 25, 3))
    # 得到需要留下的需要的index表
    for i in range(50):
        frames_left_index[i] = ((float(i + 1)) / 50.0) * float(frames_count)
        # 临近化
        if frames_left_index[i] - float(int(frames_left_index[i])) <= 0.50:
            frames_left_index[i] = float(int(frames_left_index[i]))
        else:
            frames_left_index[i] = int(frames_left_index[i] + 1)
        if frames_left_index[i] > (frames_count - 1):
            frames_left_index[i] = frames_count - 1
            if frames_count == 0:
                pass
    # 根据index表吸取对应帧到新的数组里面
    for i in range(50):
        output_frames[i] = np.array(input_frames[int(frames_left_index[i])])
    return output_frames

    # 先创建一个50位数组，算出每一位的浮点理论取帧数
    # 然后把每一位取最近值，存在另外一个数组里面


"""
source_path : 源文件地址
setup_number : 
dateCount : 对象总数
perform_id : 
filename : 从路径中提取的文件名，包括.skeleton
camera_id

get_date() : return the date(n(frame)*3(joints)*3(axis),lable
change_dir() : change source dir and refresh para
refresh_information() : refresh para so we can get the lable, performer id...

"""


class Datefile:
    dateCount = 0

    # 初始化，主要包括计数和刷新实例变量
    def __init__(self, source_path="Skeleton Based Human Action/ntu_database_part/S001C001P001R001A001.skeleton"):
        self.source_path = source_path
        # get file name
        Datefile.dateCount += 1
        self.refresh_information()
        # 单人动作和双人动作不会一起分类，先写单人的就够了
        # 还有一些特别的特征，先不考虑（如何区分参数等级，比如说重心和每个点的坐标，重心的等级是整体性的，如何比较呢？
        # 有一行是关节数，唯一变化的，可以根据这个，如果不是25，就跳过

    # 解析文件，返回numpy帧数*joint*3的数组
    def get_date(self):
        # 打开文件
        fo = open(self.source_path)
        frame_date = []
        # 读取文件，返回n（帧数）*25（joints）数组，以及lable
        print("reading file now : ", self.filename)
        # date_list = np.zeros(shape=(0, 3, 25))  # 创建总长为n的数据列
        date_list = []
        # 准备好变量
        # 逐行判断
        # 跳过开始几行
        fo.readline()
        fo.readline()
        fo.readline()
        # 开始frame内循环
        tem_frame = np.zeros(shape=(3, 0))
        frame_count = 0
        while 1:
            # 如果是指引行，保存上一帧数据，清空变量，审核当前帧joints数量，如果不是25，print打印，然后跳过行数，开启下一帧，如果是25，使劲读25行，存在temframe里
            # 判断是否是25等指引行
            # 载入新行，确定不是最后一行
            tem_line = fo.readline()
            if tem_line == "":  # 文件尾
                break
            # 如果是指引行：存数，下一步判断
            elif tem_line[0:-1].isdigit():
                tem_frame = []
                joint_count = int(tem_line[0:-1])
                # 如果joint是25：读行，数据记在tem_frame里
                if joint_count == 25:
                    for index in range(25):
                        # read 25 lines, and give back a 25*xzy number
                        str_oneline = fo.readline()
                        split_str_oneline = str_oneline.split()  # split into sigle
                        xyz_oneline = [split_str_oneline[0], split_str_oneline[1],
                                       split_str_oneline[2]]  # get the xyz
                        for str_number in range(len(xyz_oneline)):  # chage into float
                            xyz_oneline[str_number] = float(xyz_oneline[str_number])
                        tem_frame.append(xyz_oneline)
                    date_list.append(tem_frame)
                    frame_count += 1
                    # 跳过多余的两行
                    fo.readline()
                    fo.readline()
                # 如果joint不是25：跳过这些行
                else:
                    print("the", frame_count, "frame read error: not 25 !!!")
                    print("the wrong line is :", tem_line)
                    return [], []
                    # for index in range(joint_count):
                    #     fo.readline()
            last_line = tem_line
        print("date load finish!!!\n")
        # 关闭文件
        fo.close()

        return np.array(date_list), self.lable

    # 根据source_path更新当前实例变量
    def refresh_information(self):
        self.filename = self.source_path.split("/")[-1]
        # get information from file name and record #sss is the setup number, ccc is the camera ID, ppp is the performer ID, rrr is the replication number (1 or 2), and aaa is the action class label
        for index in range(len(self.filename)):
            if self.filename[index] == "S":
                self.setup_number = ""
                self.setup_number = self.filename[index] + self.filename[index + 1] + self.filename[index + 2] + \
                                    self.filename[index + 3]
            if self.filename[index] == "C":
                self.camera_id = ""
                self.camera_id = self.filename[index] + self.filename[index + 1] + self.filename[index + 2] + \
                                 self.filename[
                                     index + 3]
            if self.filename[index] == "P":
                self.perform_id = ""
                self.perform_id = self.filename[index] + self.filename[index + 1] + self.filename[index + 2] + \
                                  self.filename[index + 3]
            if self.filename[index] == "A":
                self.lable = ""
                self.lable = self.filename[index] + self.filename[index + 1] + self.filename[index + 2] + self.filename[
                    index + 3]

    # 更改路径
    def change_dir(self, source_dir):
        self.source_path = source_dir
        self.refresh_information()


"""
主程序
"""

# os.chdir("/Users/Waybaba/PycharmProjects/nturgb+d_skeletons")

#输入多个动作，返回train_date,train_lable,test_date,test_lable
def get_date(actions=["A007", "A006"]):
    # 创建空组all_date,all_lable
    all_date = []  # array也可以append，用np.append，但是numpy用的是一整块数据块，如果append会全部重新移动，所以不如先用listappend
    all_lable = []
    # 用index，遍历所有输入的标签，分别得到datelable并且apppend到all上，并且同时把lable改成0开始的整数
    for index in range(len(actions)):
        each_date, each_old_lable = get_sigle_action(actions[index])
        print(each_old_lable.shape)
        # 先把lable换成整数
        each_lable = np.zeros(shape=(len(each_old_lable)))
        for i in range(len(each_old_lable)):
            each_lable[i] = index
        # 第一次初始化一下
        if index == 0:
            all_date = each_date
            all_lable = each_lable
        else:
            all_date = np.append(all_date, each_date, axis=0)  # 如果不指定在哪个维度上append会出现
            all_lable = np.append(all_lable, each_lable, axis=0)
    all_date = np.array(all_date)
    all_lable = np.array(all_lable)
    # 打乱(保证lable,和date的打乱顺序是一样的）
    state = np.random.get_state()
    np.random.shuffle(all_date)
    np.random.set_state(state)  # 状态决定顺序
    np.random.shuffle(all_lable)
    # 划分训练和测试并返回
    num_of_date = all_date.shape[0]
    train_num = int((4*num_of_date) / 5)
    return all_date[0:train_num],all_lable[0:train_num] , all_date[train_num:],all_lable[train_num:]


# 输入单个action编码，返回date和lable
def get_sigle_action(action="A007", folder_dir="/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/nturgb+d_skeletons/"):
    os.chdir(folder_dir)
    filename_list = []
    datefile = Datefile()
    datelist_buf = []  # 暂存
    lablelist_buf = []
    for each_filename in os.listdir():
        if each_filename[16:20] == action:
            filename_list.append(each_filename)
    error_load_date_count = 0  # discard date count
    for each_filename in filename_list:
        file_path = os.path.join(folder_dir, each_filename)
        datefile.change_dir(file_path)
        date_buf, lable_buf = datefile.get_date()
        if date_buf == []:  # 防止错误
            error_load_date_count += 1
        else:
            datelist_buf.append(frame_cut(date_buf))
            lablelist_buf.append(lable_buf)
    date = np.array(datelist_buf)
    lable = np.array(lablelist_buf)
    print("Loading date for action " + lable[0] + " finish!!!\n(Discard %d dates because of loading error.)" % error_load_date_count)
    print("Whole date shape is :", end="")
    print(date.shape)
    return date, lable

#使用pickle来存取date
def make_date():
    save_name = "40_actions"
    save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
    if not os.path.exists(save_path + save_name):
        os.makedirs(save_path + save_name)
        # 下面是要存的数据
        input_train, y_train, input_test, y_test = ntu.get_date(
            ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008", "A009", "A010", "A011", "A012", "A013",
             "A014", "A015", "A016", "A017", "A018", "A019", "A020", "A021", "A022", "A023", "A024", "A025", "A026",
             "A027", "A028", "A029", "A030", "A031", "A032", "A033", "A034", "A035", "A036", "A037", "A038", "A039",
             "A040"])
        # open file and save, creat file variable, then dump date into the file
        f = open(save_path + save_name + '/' + "input_train.txt", 'wb')
        pickle.dump(input_train, f)
        f.close()
        f = open(save_path + save_name + '/' + "y_train.txt", 'wb')
        pickle.dump(y_train, f)
        f.close()
        f = open(save_path + save_name + '/' + "input_test.txt", 'wb')
        pickle.dump(input_test, f)
        f.close()
        f = open(save_path + save_name + '/' + "y_test.txt", 'wb')
        pickle.dump(y_test, f)
        f.close()
def load_date(save_name):
    #先写一个两个提取的
    #再写一个3个提取的
    #之后再考虑要不要把所有的单独提出来
    if save_name=="2_actions":
        save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
        if not os.path.exists(save_path + save_name):
            os.makedirs(save_path + save_name)
            input_train, y_train, input_test, y_test = get_date(["A007", "A006"])
            # open file and sava
            f = open(save_path + save_name +'/'+"input_train.txt", 'wb')
            pickle.dump(input_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'wb')
            pickle.dump(y_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'wb')
            pickle.dump(input_test, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'wb')
            pickle.dump(y_test, f)
            f.close()
        else:
            f = open(save_path + save_name + '/' + "input_train.txt", 'rb')
            input_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'rb')
            y_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'rb')
            input_test = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'rb')
            y_test = pickle.load(f)
            f.close()
    elif save_name=="4_actions":
        save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
        if not os.path.exists(save_path + save_name):
            os.makedirs(save_path + save_name)
            input_train, y_train, input_test, y_test = get_date(["A004", "A005","A006","A007"])
            # open file and sava
            f = open(save_path + save_name + '/' + "input_train.txt", 'wb')
            pickle.dump(input_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'wb')
            pickle.dump(y_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'wb')
            pickle.dump(input_test, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'wb')
            pickle.dump(y_test, f)
            f.close()
        else:
            f = open(save_path + save_name + '/' + "input_train.txt", 'rb')
            input_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'rb')
            y_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'rb')
            input_test = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'rb')
            y_test = pickle.load(f)
            f.close()
    elif save_name=="40_actions":
        save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
        if not os.path.exists(save_path + save_name):
            os.makedirs(save_path + save_name)
            input_train, y_train, input_test, y_test = get_date(["A004", "A005", "A006", "A007"])
            # open file and sava
            f = open(save_path + save_name + '/' + "input_train.txt", 'wb')
            pickle.dump(input_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'wb')
            pickle.dump(y_train, f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'wb')
            pickle.dump(input_test, f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'wb')
            pickle.dump(y_test, f)
            f.close()
        else:
            f = open(save_path + save_name + '/' + "input_train.txt", 'rb')
            input_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_train.txt", 'rb')
            y_train = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "input_test.txt", 'rb')
            input_test = pickle.load(f)
            f.close()
            f = open(save_path + save_name + '/' + "y_test.txt", 'rb')
            y_test = pickle.load(f)
            f.close()
    else : return 0
    input_train = input_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    input_test = input_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return input_train,y_train,input_test,y_test
def change_into_muti_dim(input):
    #得到最大值，推出分类数，得到长度，决定目标长
    kinds = input.max()+1
    date_length = len(input)
    #创建空数组0，float32
    target = np.zeros(shape=[date_length, int(kinds)], dtype=float)
    #index遍历，根据数字给指定位赋值1
    for index in range(date_length):
        type = input[index]
        target[index][int(type)] = 1.0
    #返回
    return target



# get filename of throw/pickup
# throw_filename_list = []
# pickup_filename_list = []
# dir_list = os.listdir()
# datefile = Datefile()
# for each_filename in os.listdir():
#     action = each_filename[16:20]
#     if each_filename[16:20] == "A006":
#         throw_filename_list.append(each_filename)
#     elif each_filename[16:20] == "A007":
#         pickup_filename_list.append(each_filename)
# #拼接添加到date里面
# train_date_buf = []
# train_lable_buf = []
# error_load_date_count = 0 #discard date count
# for each_pickup_filename in pickup_filename_list :
#     file_path=os.path.join("/Users/Waybaba/PycharmProjects/nturgb+d_skeletons/",each_pickup_filename)
#     datefile.change_dir(file_path)
#     date_buf , lable_buf = datefile.get_date()
#     if date_buf ==[]:#防止错误
#         error_load_date_count += 1
#     else :
#         train_date_buf.append(frame_cut(date_buf))
#         train_lable_buf.append(lable_buf)
# train_date = np.array(train_date_buf)
# train_lable = np.array(train_lable_buf)
# # train_date_shape = train_date.shape
# print("Whole date shape is :",end="")
# print(train_date.shape)
# print("Loading date for action "+train_lable[0]+" finish!!!\n(Discard %d dates because of loading error.)" % error_load_date_count)
# for each_lable in train_lable:


##尝试分函数进行date的制作
# throw_date , throw_lable = get_sigle_action(action="A006")
# pickup_date , pickup_lable = get_sigle_action(action= "A007")
# all_date , all_lable = []
# all_date.append(throw_date)
# all_date.append(pickup_date)
# all_lable.append(throw_date,pickup_date)
# mix_dif_action_date(all_date,all_lable)

#最后的测试代码
# train_date,train_lable,test_date,test_lable = get_date(["A007","A006"])
# print(train_date.shape)
# print(train_lable.shape)
# print(test_date.shape)
# print(test_lable.shape)