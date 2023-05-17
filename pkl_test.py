import pickle
import numpy as np
with open('./data/sensor_graph/adj_mx_bay (1).pkl','rb') as f:
    data = pickle.load(f,encoding='latin1')

data = np.array(data,dtype=object)
data = data[2]
non_zero_count = np.count_nonzero(data)

# 打印非零元素的数量
print(non_zero_count)
print(data.shape)

