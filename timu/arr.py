# -- coding: utf-8 --
# mnist attention
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import pandas as pd
import keras
# from keras.utils import plot_model
from keras.layers import Input, Embedding, LSTM, Dense,Bidirectional


TIME_STEPS_1 = 10
TIME_STEPS_2=3
lstm_units = 128


main_x=pd.read_csv('train_para.csv')
main_x=main_x.values

main_y=pd.read_csv('train_label.csv')
main_y=main_y.values
y_train=main_y
y_train =np_utils.to_categorical(main_y,num_classes=2)

add_x=pd.read_csv('train_ques.csv')
add_x=add_x.values


# print(x.shape)

# first way attention
def attention_3d_block(inputs,TIME_STEPS):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# build RNN model with attention
#问题输入，LSTM+attention
main_input = Input(shape=(len(main_x[0]),), dtype='float32', name='main_input')
train_x = Embedding(output_dim=512, input_dim=len(main_x), input_length=len(main_x[0]))(main_input)
lstm_out_1 = Bidirectional(GRU(128, return_sequences=True),merge_mode='concat')(train_x)
lstm_out_2 = Bidirectional(GRU(40, return_sequences=True),merge_mode='concat')(lstm_out_1)

attention_mul = attention_3d_block(lstm_out_1,TIME_STEPS_1)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.2)(attention_flatten)

#另一端输入 lstm
add_x=pd.read_csv('train_ques.csv')
add_x=add_x.values
auxiliary_input = Input(shape=(len(add_x[0]),), name='aux_input')
auxiliary_input_x=Embedding(output_dim=512, input_dim=len(add_x), input_length=len(add_x[0]))(auxiliary_input)
lstm_out_aux = Bidirectional(GRU(64, return_sequences=False), name='bilstm')(auxiliary_input_x)
drop2_aux = Dropout(0.3)(lstm_out_aux)


x = keras.layers.concatenate([drop2, drop2_aux])
#全连接

x_out_1 = Dense(20, activation='relu')(x)
# x_out = Dense(60, activation='relu')(x_out_1)
output = Dense(2, activation='sigmoid')(x_out_1)

model = Model(inputs=[main_input,auxiliary_input], outputs=output)

# model = Model(inputs=[main_input,auxiliary_input], outputs=[output])

adam=Adam(lr=1e-3)
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])


# plot_model(model, 'Multi_input.png')

print(model.summary())

print('Training------------')

model.fit([main_x, add_x], y_train, validation_split=0.1,epochs=5, batch_size=32,verbose=1)
