import os

import numpy as np
import pandas as pd

# from_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/br'
# to_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/br4kernel'
from_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/NB_notcol'
to_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/NB_notcol_avg'

# 目的： 把每个from的xlsx数据求个均值扔到to里面

raw_list = os.listdir(from_path)

for idx, xlsx in enumerate(raw_list):
    source = os.path.join(from_path, xlsx)  # 1 fill in the source path
    data = pd.read_excel(source).to_numpy()  # 2 read data
    new_data = np.average(data, axis=1)  # 3 calculate new data

    new_path = os.path.join(to_path, xlsx)  # 4 fill in new path with old name
    pd.DataFrame(new_data).to_excel(new_path, index=False)  # 5 save new data to new path
    print(idx, xlsx)
