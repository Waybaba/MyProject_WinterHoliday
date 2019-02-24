# encoding: utf-8
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
import numpy as np
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错



def change_into_predict_data(video_data,frames_length=3):
    total_video_num = video_data.shape[0]
    result_frames_list = []
    result_label_list = []
    for video_index in range(total_video_num):
        current_video = video_data[video_index]
        for frame_index in range(50 - frames_length):
            result_frames_list.append(current_video[frame_index:frame_index + frames_length])
            result_label_list.append(current_video[frame_index + frames_length])
    result_frames_list = np.array(result_frames_list)
    result_label_list = np.array(result_label_list)
    print(result_frames_list.shape)
    print(result_label_list.shape)
    return result_frames_list, result_label_list


if __name__ == "__main__":
    # load_date
    input_train,lable_train,input_test,lable_test =ntu.load_date("40_actions")
    # ntusee.show_gif(input_train[1])
    input_train = input_train.reshape((-1,50,75))#数据整形，然后输
    input_test = input_test.reshape((-1,50,75))
    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)

    frames_list,label_list = change_into_predict_data(input_train)






    # save frames date, label