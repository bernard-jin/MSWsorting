import os
import shutil

normal_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/NotBlack'
black_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Data3/HeFei/xlsx/black'

for item in os.listdir(normal_path):
    if 'b' in item:
        shutil.move(os.path.join(normal_path, item), black_path)
        print(item)
