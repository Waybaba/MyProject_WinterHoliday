# encoding: utf-8
from is_for_import.finish_env import frameChooseEnv
from is_for_import.DQN_test import DQN
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import os,time
import numpy as np


a = np.random.random(size=(5))
b = np.argsort(a)
b_max = b[-3:]
index_01 = np.zeros(shape=(5))
index_01[b_max] = 1.0
result = np.empty(shape=(3))
index_record = 0
for i in range(5):
    if index_01[i] == 1.0:
        result[index_record] = a[i]
        index_record += 1
print(a)
print(result)