# encoding: utf-8
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import os,time
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from scipy import optimize
from keras.utils import plot_model
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错
from scipy.interpolate import spline

def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D
def show_activity_rate(data,plot=False):
    data_length = data.shape[0]
    activity = np.zeros(shape=(data_length))
    activity[0] = ((data[0] - data[1]) ** 2).mean(axis=None)
    activity[-1] = ((data[-1] - data[-2]) ** 2).mean(axis=None)
    for i in range(data_length)[2:-2]:
        activity[i] = ((data[i] - data[i-1]) ** 2).mean(axis=None)
    if plot :
        x = np.arange(data_length)
        plt.ylim(ymax=0.01)
        plt.plot(x, activity, 'bo-')
        plt.title('Activity - Frames')
        plt.xlabel('frames index')
        plt.ylabel('activity')
        plt.legend()
        plt.show()
    return activity



if __name__ == "__main__":
    # load_date
    input_train, lable_train, input_test, lable_test = ntu.load_date("4_actions")
    # ntusee.show_gif(input_train[0])
    input_train = input_train.reshape((-1, 50, 75))  # 数据整形，然后输
    input_test = input_test.reshape((-1, 50, 75))

    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)

    model = models.load_model(filepath="model_backup/predict_net.h5")

    #
    old_sequence = input_train[13]
    # ntusee.show_gif(np.reshape(a=old_sequence,newshape=(50,25,3)),time_interval=0.05)
    # ntusee.show_gif(np.reshape(a=np.array([old_sequence[0],old_sequence[10],old_sequence[20],old_sequence[30],old_sequence[40]]), newshape=(-1, 25, 3)), time_interval=0.5)

    predict_sequence = np.empty(shape=(50,75))
    predict_sequence[0] = old_sequence[0]
    predict_sequence[1] = old_sequence[1]
    for i in range(48):
        # predict_sequence[i+2]=model.predict(x=predict_sequence[np.newaxis,i:i+2])
        predict_sequence[i + 2] = model.predict(x=old_sequence[np.newaxis, i:i + 2])
    mse = ((predict_sequence - old_sequence) ** 2).mean(axis=1)
    new_mse = np.zeros(shape=mse.shape)
    for i in range(50)[2:-2]:
        new_mse[i]=(mse[i-1]+mse[i]+mse[i+1])/3
    # mse = new_mse
    predict_sequence = np.reshape(predict_sequence,newshape=(50,25,3))
    # ntusee.show_gif(predict_sequence,time_interval=0.5)

    x = np.arange(50)

    A,B,C,D = optimize.curve_fit(f_3, x, mse)[0]
    x_line = np.arange(0,50,0.1)
    y_line = np.zeros(shape=(x_line.shape[0]))
    for i in range(x_line.shape[0]):
        y_line[i] = f_3(x_line[i],A,B,C,D)
    # l3 = plt.plot(x, mse, 'b--', label='type3')

    xnew = np.linspace(x.min(), x.max(), 300)  # 300 represents number of points to make between T.min and T.max

    power_smooth = spline(x, mse, xnew)

    plt.ylim(ymax=0.02)
    plt.plot(x, mse, 'r-',label="loss")
    # plt.plot(xnew, power_smooth, 'b-', label='loss')
    plt.plot(x,show_activity_rate(old_sequence),"g-",label="activity")
    plt.title('Loss - Frames')
    plt.xlabel('frames index')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

