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


main_x=pd.read_csv('train_para.csv',header = None)
main_x=main_x.values
# main_x=np.array(main_x)
# main_x = np.random.random((10000,100))
main_y=pd.read_csv('train_label.csv',header = None)
main_y=main_y.values
# main_y = keras.utils.to_categorical(np.random.randint(1,size = (10000,1)))

# main_y = np.random.randint(0, 2, (100,))
#additional_data
'''
All input arrays (x) should have the same number of samples. Got array shapes:
主数据集和额外的数据集的输入的特征张量的数据集个数相等，也就是行数相等；
'''
add_x=pd.read_csv('train_ques.csv',header = None)
add_x=add_x.values
# add_x=np.array(add_x)
# add_x = np.random.random((10000,100))

'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
main_input_1 = Input(shape=(len(main_x[0])+1,), dtype='float32', name='main_input_1')
# auxiliary_input = Input(shape=(len(add_x[0]),), name='aux_input')
x_1 = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0])+1)(main_input_1)
lstm_out_1 = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(x_1)

main_input_2 = Input(shape=(len(main_x[0])+1,), dtype='float32', name='main_input_2')
x_2 = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0])+1)(main_input_2)
lstm_out_2 = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(x_2)

main_input_3 = Input(shape=(len(main_x[0])+1,), dtype='float32', name='main_input_3')
x_3 = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0])+1)(main_input_3)
lstm_out_3 = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(x_3)

# lstm_out = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(lstm_out)


#额外的输入数据
'''
所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
'''
# auxiliary_input = Input(shape=(len(add_x[0]),), name='aux_input')

# auxiliary_input_x=Embedding(output_dim=512, input_dim=len(add_x), input_length=len(add_x[0]))(auxiliary_input)

# lstm_out_aux = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(auxiliary_input_x)
# lstm_out_aux = Bidirectional(LSTM(128,return_sequences=False),merge_mode='concat')(lstm_out_aux)


'''
#将LSTM得到的张量与额外输入的数据串联起来，这里是横向连接
'''
x = keras.layers.concatenate([lstm_out_1, lstm_out_2,lstm_out_3])

#建立一个深层连接的网络
# We stack a deep densely-connected network on top
x = Dense(128, activation='relu')(x)

#得到主数据集输出的张量，10与输入的主数据集标签数据集的标签类相等
# And finally we add the main logistic regression layer

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
# main_output = Dense(2, activation='softmax', name='main_output')(x)

#生成模型
model = Model(inputs=[main_input_1, main_input_2,main_input_3], outputs=[main_output])
#plot_model(model, 'Multi_input.png')
#编译模型

adam=Adam(lr=1e-3)
model.compile(optimizer=adam, loss='binary_crossentropy',loss_weights=[1],metrics=['accuracy'])
# model.compile(optimizer=adam, loss='categorical_crossentropy',loss_weights=[2],metrics=['accuracy'])
#训练模型l
aa=add_x[:,0:1]
aa = aa.reshape(-1, 1)
bb=add_x[:,1:2]
bb= bb.reshape(-1, 1)
cc=add_x[:,2:3]
cc= cc.reshape(-1, 1)

main_x_1=np.concatenate((main_x,aa), axis= 1)
main_x_2=np.concatenate((main_x,bb),axis=1)
main_x_3=np.concatenate((main_x,cc),axis=1)


# model.fit([main_x, auxiliary_input], [main_y],validation_split=0.33,epochs=5, batch_size=32,verbose=1)
model.fit([main_x_1, main_x_2,main_x_3], [main_y],validation_split=0.33,epochs=5, batch_size=32,verbose=1)
#yuce
test_p=pd.read_csv('test_para.csv',header = None)
test_p=test_p.values
test_q=pd.read_csv('test_ques.csv',header = None)
test_q=test_q.values

a=test_q[:,0:1]
a = a.reshape(-1, 1)
b=test_q[:,1:2]
b= b.reshape(-1, 1)
c=test_q[:,2:3]
c= c.reshape(-1, 1)

test_p_1=np.concatenate((test_p,a), axis= 1)
test_p_2=np.concatenate((test_p,b),axis=1)
test_p_3=np.concatenate((test_p,c),axis=1)

classes=model.predict([test_p_1,test_p_2,test_p_3],batch_size=1)
# print(classes)
classed = [np.round(x) for x in classes]

# for i in classes:
#     if classes[i]>0.5:
#         classes[i]=1
#     else:
#         classes[i]=0
# print(int(classes[:]))
# classes=np.array(classes,dtype=int)

# print(classes)
data1 = pd.DataFrame(classed)
data1.to_csv('test_pred.csv',header=0, index=0)


# n_in_timestep=1
# model.save('./model/my_model_combine_timestep_LSTM%s_1000days_0429.h5' % n_in_timestep)
model.save('./model/my_model_combine_timestep_LSTM_1000days_0430.h5')
