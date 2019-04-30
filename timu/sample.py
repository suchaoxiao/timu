# -- coding: utf-8 --
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:28:27 2018
函数模型之多输入与多输出模型
@author: BruceWong
"""
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
#generate data
#main_data
#这里构建数据集：主数据集为10000*100的二维数组，意味着100个特征
#标签为10000*1的二维数组，共有10种输出结果
main_x = np.random.random((10000,100))
main_y = keras.utils.to_categorical(np.random.randint(10,size = (10000,1)))
#additional_data
'''
All input arrays (x) should have the same number of samples. Got array shapes:
主数据集和额外的数据集的输入的特征张量的数据集个数相等，也就是行数相等；
'''
add_x = np.random.random((10000,10))
add_y = keras.utils.to_categorical(np.random.randint(10,size = (10000,1)))
# 设定主要输入的张量，并命名main_input
# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
main_input = Input(shape=(100,), dtype='int32', name='main_input')
# 嵌入生成512列维度的张量
# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
#print(x.shape)
#使用LSTM模型
# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
#额外的输入数据
'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
auxiliary_input = Input(shape=(10,), name='aux_input')

'''
#将LSTM得到的张量与额外输入的数据串联起来，这里是横向连接
'''
x = keras.layers.concatenate([lstm_out, auxiliary_input])
#建立一个深层连接的网络
# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#得到主数据集输出的张量，10与输入的主数据集标签数据集的标签类相等
# And finally we add the main logistic regression layer
main_output = Dense(10, activation='sigmoid', name='main_output')(x)

#生成模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
#编译模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[1., 0.2])
#训练模型
model.fit([main_x, add_x], [main_y, main_y],epochs=10, batch_size=128)

#model.compile(optimizer='rmsprop',loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
#model.fit({'main_input': headline_data, 'aux_input': additional_data},{'main_output': labels, 'aux_output': labels},epochs=50, batch_size=32)
