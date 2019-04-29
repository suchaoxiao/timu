# -- coding: utf-8 --
import csv
import os
import numpy as np

'''将iris.csv中的数据分成train_iris和test_iris两个csv文件，其中train_iris.csv中有120个数据，test_iris.csv中有30个数据'''
labels = []
data = []
a_train_file = 'train_iris.csv'
a_test_file = 'test_iris.csv'
a_file = 'iris.csv'

seed = 3
np.random.seed(seed)
train_indices = np.random.choice(150, 120, replace=False) # 设置随机数生成从0-150中随机挑选120个随机数
residue = np.array(list(set(range(150)) - set(train_indices)))
test_indices = np.random.choice(len(residue),30, replace=False) # 如果训练集和测试集综合的数据加起来就是一整个数据集则不需要这个操作

with open(a_file)as afile:
    a_reader = csv.reader(afile)  #从原始数据集中将所有数据读取出来并保存到a_reader中
    labels = next(a_reader)  # 提取第一行设置为labels
    for row in a_reader:  # 将a_reader中每一行的数据提取出来并保存到data的列表中
        data.append(row)


# 生成训练数据集
if not os.path.exists(a_train_file):
    with open(a_train_file, "w", newline='') as a_trian:
        writer = csv.writer(a_trian)
        writer.writerows([labels])  #第一行为标签行
        writer.writerows(np.array(data)[train_indices])
        a_trian.close()

# 生成测试数据集
if not os.path.exists(a_test_file):
    with open(a_test_file, "w", newline='')as a_test:
        writer = csv.writer(a_test)
        writer.writerows([labels])  #第一行为标签行
        writer.writerows(np.array(data)[test_indices])
        a_test.close()
