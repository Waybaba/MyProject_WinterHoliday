import numpy as np
import math
distance  = 10

whole_series  = np.zeros(shape=[distance+2])
whole_series[0] = 1
whole_series[-1] = 1

print(whole_series)

middle = (0 + distance+1)/2
down = math.ceil(middle)-1
up = math.ceil(middle)


whole_series_new  = np.zeros(shape=[distance+2])
whole_series_new[]

