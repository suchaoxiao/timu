# -- coding: utf-8 --

"""
Created on Fri Jan 12 10:28:27 2018
函数模型之多输入与多输出模型
@author: BruceWong
"""
import keras
from keras.layers import Input, Embedding, LSTM, Dense,Bidirectional
from keras.models import Model
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.optimizers import Adam


main_x=pd.read_csv('train_para.csv')
main_x=main_x.values
# main_x = np.random.random((10000,100))
main_y=pd.read_csv('train_label.csv')
main_y=main_y.values
# main_y = keras.utils.to_categorical(np.random.randint(1,size = (10000,1)))

# main_y = np.random.randint(0, 2, (100,))
#additional_data
'''
All input arrays (x) should have the same number of samples. Got array shapes:
主数据集和额外的数据集的输入的特征张量的数据集个数相等，也就是行数相等；
'''

add_x=pd.read_csv('train_ques.csv')
add_x=add_x.values
# add_x = np.random.random((10000,100))

'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
main_input = Input(shape=(len(main_x[0]),), dtype='float32', name='main_input')
x = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0]))(main_input)
lstm_out = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(x)
# lstm_out = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(lstm_out)


#额外的输入数据
'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
auxiliary_input = Input(shape=(len(add_x[0]),), name='aux_input')

auxiliary_input_x=Embedding(output_dim=512, input_dim=len(add_x), input_length=len(add_x[0]))(auxiliary_input)

lstm_out_aux = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(auxiliary_input_x)
# lstm_out_aux = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(lstm_out_aux)


'''
#将LSTM得到的张量与额外输入的数据串联起来，这里是横向连接
'''
x = keras.layers.concatenate([lstm_out, lstm_out_aux])

#建立一个深层连接的网络
# We stack a deep densely-connected network on top
x = Dense(128, activation='relu')(x)

#得到主数据集输出的张量，10与输入的主数据集标签数据集的标签类相等
# And finally we add the main logistic regression layer

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

#生成模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
#plot_model(model, 'Multi_input.png')
#编译模型

adam=Adam(lr=1e-3)
model.compile(optimizer=adam, loss='binary_crossentropy',loss_weights=[1],metrics=['accuracy'])
#训练模型l
model.fit([main_x, add_x], [main_y],validation_split=0.33,epochs=50, batch_size=32,verbose=1)

n_in_timestep=1
model.save('./model/my_model_combine_timestep_LSTM%s_1000days_0429.h5' % n_in_timestep)
