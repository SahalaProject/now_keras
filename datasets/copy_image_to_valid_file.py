import os
import shutil


"""
拆分数据集，分类拷贝
"""

with open('./food-101/meta/classes.txt', 'r')as fp:
    names = fp.read()

class_names = names.split('\n')
print(class_names)


data = './food-101/images/'

for class_name in class_names:
    class_data = os.path.join(data, class_name) # 对于路径下循环到的class_data
    # print(class_data)
    if not class_data.endswith('images/'):
        for root, dir, files in os.walk(class_data):
            print(root)
            # print(dir)
            # print(file)  # list
            # 拷贝部分到验证集，，身下的作为训练集
            valid_file = os.path.join('./food-101/validation/', class_name)
            valid_file if os.path.exists(valid_file) else os.mkdir(valid_file)

            # 拷贝该类型下的一定数量的file
            file_num = 0
            for file in files:
                file_num+=1
                if file_num <= 200:
                    shutil.move(os.path.join(root, file), os.path.join(valid_file, file))
                else:
                    break
