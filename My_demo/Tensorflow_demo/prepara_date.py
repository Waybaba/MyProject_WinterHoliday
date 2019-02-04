from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import random
import pickle
import is_for_import.ntu_date_preprocess_2 as ntu

save_name = "40_actions"
save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
if not os.path.exists(save_path + save_name):
    os.makedirs(save_path + save_name)
    input_train, y_train, input_test, y_test = ntu.get_date(["A001","A002","A003","A004","A005","A006","A007","A008","A009","A010","A011","A012","A013","A014","A015","A016","A017","A018","A019","A020","A021","A022","A023","A024","A025","A026","A027","A028","A029","A030","A031","A032","A033","A034","A035","A036","A037","A038","A039","A040"])
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

# for i in range(40):
#     print('\"',end='')
#     print('A0',end='')
#     if i < 10 :
#         print('0',end='')
#     print(i,end='')
#     print('\",',end='')


