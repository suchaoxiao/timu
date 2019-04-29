# -- coding: utf-8 --
from keras.layers import Conv1D, Dense, MaxPool1D, concatenate, Flatten,Embedding,LSTM
from keras import Input, Model
from keras.utils import plot_model
import numpy as np


def multi_input_model():
    """构建多输入模型"""
    main_input = Input(shape=(1,), dtype='int32', name='main_input')
    # 嵌入生成512列维度的张量
    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=32, input_dim=7, input_length=1)(main_input)
    # print(x.shape)
    # 使用LSTM模型
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)
    auxiliary_output = Dense(10, activation='sigmoid', name='aux_output')(lstm_out)

    # 额外的输入数据
    '''
    所有的input里面的shape的维度均是特征个数维度，列的个数；shape=(特征个数,)
    '''
    auxiliary_input = Input(shape=(1,), name='aux_input')
    '''
    #将LSTM得到的张量与额外输入的数据串联起来，这里是横向连接
    '''
    q = Embedding(output_dim=32, input_dim=3, input_length=1)(auxiliary_input)
    x = concatenate([lstm_out,q])
    # 建立一个深层连接的网络
    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)


    # 得到主数据集输出的张量，10与输入的主数据集标签数据集的标签类相等
    # And finally we add the main logistic regression layer
    main_output = Dense(2, activation='sigmoid', name='main_output')(x)

    # 生成模型
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    # 编译模型
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
    # 训练模型
    model.fit([main_x, add_x], [main_y, main_y], epochs=10, batch_size=128)





    input1_ = Input(shape=(100, 1), name='input1')
    input2_ = Input(shape=(50, 1), name='input2')


    x1 = Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input1_)
    x1 = MaxPool1D(pool_size=10, strides=10)(x1)

    x2 = Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=5, strides=5)(x2)

    x = concatenate([x1, x2])
    x = Flatten()(x)

    x = Dense(10, activation='relu')(x)
    output_ = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[input1_, input2_], outputs=[output_])
    model.summary()

    return model


if __name__ == '__main__':
    # 产生训练数据
    x1 = np.random.rand(100, 100, 1)
    x2 = np.random.rand(100, 50, 1)
    # 产生标签
    y = np.random.randint(0, 2, (100,))

    model = multi_input_model()
    # 保存模型图
    plot_model(model, 'Multi_input_model.png')

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit([x1, x2], y, epochs=5, batch_size=32)
