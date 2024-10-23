import pandas as pd
import shutil
import os
import openpyxl

normal_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/NotBlack'
black_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/black'
cleandata_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/NB_notcol'

for item in os.listdir(normal_path):
    data = pd.read_excel(os.path.join(normal_path, item), header=1)

    data = data.drop(columns=877.250000)
    data.to_excel(os.path.join(cleandata_path, item), index=False)
    print(item)