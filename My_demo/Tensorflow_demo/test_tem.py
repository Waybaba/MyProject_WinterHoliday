import numpy as np
import math


choose_index_50_to_30 = np.arange(30)
for i in np.arange(30):
    choose_index_50_to_30[i] = math.floor(choose_index_50_to_30[i] * 5/3)
print(choose_index_50_to_30)