import tensorflow as tf
from tensorflow import keras
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn import preprocessing
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def model_wavenet_timeseries(len_seq, len_out, dim_exo, nb_filters, dim_filters, dilation_depth, use_bias, res_l2,
                             final_l2, batch_size):
    input_shape_exo = tf.keras.Input(shape=(len_seq, dim_exo), name='input_exo')
    input_shape_target = tf.keras.Input(shape=(len_seq, 1), name='input_target')

    input_exo = input_shape_exo
    input_target = input_shape_target

    outputs = []

    # 1st loop: I'll be making multi-step predictions (every prediction will serve as input for the second prediction, when I move the window size...).
    for t in range(len_out):

        # Causal convolution for the 1st input: exogenous series
        # out_exo : (batch_size,len_seq,dim_exo)
        out_exo = K.temporal_padding(input_exo,
                                     padding=((dim_filters - 1), 0))  # To keep the same len_seq for the hidden layers
        out_exo = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=dim_filters, dilation_rate=1, use_bias=True,
                                         activation=None,
                                         name='causal_convolution_exo_%i' % t,
                                         kernel_regularizer=tf.keras.regularizers.l2(res_l2))(out_exo)
        # Causal convolution for the 2nd input: target series
        # out_target : (batch_size,len_seq,1)
        out_target = K.temporal_padding(input_target, padding=(
        (dim_filters - 1), 0))  # To keep the same len_seq for the hidden layers
        out_target = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=dim_filters, dilation_rate=1, use_bias=True,
                                            activation=None,
                                            name='causal_convolution_target_%i' % t,
                                            kernel_regularizer=tf.keras.regularizers.l2(res_l2))(out_target)
        # Dilated Convolution (1st Layer #1): part of exogenous series(see visual above)
        # skip_exo : (batch_size,len_seq,dim_exo)
        # first_out_exo (the same as skip_exo, before adding the residual connection): (batch_size, len_seq, dim_exo)
        skip_outputs = []
        z_exo = K.temporal_padding(out_exo, padding=(2 * (dim_filters - 1), 0))
        z_exo = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=dim_filters, dilation_rate=2, use_bias=True,
                                       activation="relu",
                                       name='dilated_convolution_exo_1_%i' % t,
                                       kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_exo)

        skip_exo = tf.keras.layers.Conv1D(dim_exo, 1, padding='same', use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_exo)
        first_out_exo = tf.keras.layers.Conv1D(dim_exo, 1, padding='same', use_bias=False,
                                               kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_exo)

        # Dilated Convolution (1st Layer #1): part of target series(see visual above)
        # skip_target : (batch_size,len_seq,1)
        # first_out_target (the same as skip_target, before adding the residual connection): (batch_size, len_seq,1)
        z_target = K.temporal_padding(out_target, padding=(2 * (dim_filters - 1), 0))
        z_target = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=dim_filters, dilation_rate=2, use_bias=True,
                                          activation="relu",
                                          name='dilated_convolution_target_1_%i' % t,
                                          kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_target)
        skip_target = tf.keras.layers.Conv1D(1, 1, padding='same', use_bias=False,
                                             kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_target)
        first_out_target = tf.keras.layers.Conv1D(1, 1, padding='same', use_bias=False,
                                                  kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z_target)

        # Concatenation of the skip_exo & skip_target AND Saving them as we go through the 1st for loop (through len_seq)
        # skip_exo : (batch_size,len_seq,dim_exo)
        # skip_target : (batch_size,len_seq,1)
        # The concatenation of skip_exp & skip_target : (batch_size,len_seq,dim_exo+1)
        skip_outputs.append(tf.concat([skip_exo, skip_target], axis=2))
        # Adding the Residual connections of the exogenous series & target series to inputs
        # res_exo_out : (batch_size,len_seq,dim_exo)
        # res_target_out : (batch_size,len_seq,1)
        res_exo_out = tf.keras.layers.Add()([input_exo, first_out_exo])
        res_target_out = tf.keras.layers.Add()([input_target, first_out_target])

        # Concatenation of the updated outputs (after having added the residual connections)
        # out_concat : (batch_size,len_seq,dim_exo+1)
        out_concat = tf.concat([res_exo_out, res_target_out], axis=2)
        out = out_concat

        # From 2nd Layer Layer#2 to Final Layer (Layer #L): See Visual above
        # 2nd loop: inner loop (Going through all the intermediate layers)
        # Intermediate dilated convolutions
        for i in range(2, dilation_depth + 1):
            z = K.temporal_padding(out_concat, padding=(2 ** i * (dim_filters - 1), 0))  # To keep the same len_seq
            z = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=dim_filters, dilation_rate=2 ** i, use_bias=True,
                                       activation="relu",
                                       name='dilated_convolution_%i_%i' % (i, t),
                                       kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z)

            skip_x = tf.keras.layers.Conv1D(dim_exo + 1, 1, padding='same', use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z)
            first_out_x = tf.keras.layers.Conv1D(dim_exo + 1, 1, padding='same', use_bias=False,
                                                 kernel_regularizer=tf.keras.regularizers.l2(res_l2))(z)
            res_x = tf.keras.layers.Add()([out, first_out_x])  # (batch_size,len_seq,dim_exo+1)
            out = res_x
            skip_outputs.append(skip_x)  # (batch_size,len_seq,dim_exo+1)

        # Adding intermediate outputs of all the layers (See Visual above)
        out = tf.keras.layers.Add()(skip_outputs)  # (batch_size,len_seqe,dim_exo+1)

        # Output Layer (exogenous series & target series)
        # outputs : (batch_size,1,1) # The predcition Y
        # out_f_exo (will be used to Adjust the input for the following prediction, see below) : (batch_size,len_seq,dim_exo)
        out = tf.keras.layers.Activation('linear', name="output_linear_%i" % t)(out)
        out_f_target = tf.keras.layers.Conv1D(1, 1, padding='same',
                                              kernel_regularizer=tf.keras.regularizers.l2(final_l2))(out)
        out_f_exo = tf.keras.layers.Conv1D(dim_exo, 1, padding='same',
                                           kernel_regularizer=tf.keras.regularizers.l2(final_l2))(out)
        outputs.append(out_f_target[:, -1:, :])

        # Adjusting the entry of exogenous series,for the following prediction (of MultiStep predictions): see the 1st loop
        # First: concatenate the previous exogenous series input with the final exo_prediction(current exo_prediction)
        input_exo = tf.concat([input_exo, out_f_exo[:, -1:, :]], axis=1)
        # Second, shift the moving window (of len_seq size) by one
        input_exo = tf.slice(input_exo, [0, 1, 0], [batch_size, input_shape_exo.shape[1], dim_exo])

        # Adjusting the entry of target series,for the following prediction (of MultiStep predictions): see the 1st loop
        # First: concatenate the previous target series input with the final prediction(current prediction)
        input_target = tf.concat([input_target, out_f_target[:, -1:, :]], axis=1)
        # Second, shift the moving window (of len_seq size) by one
        input_target = tf.slice(input_target, [0, 1, 0], [batch_size, input_shape_target.shape[1], 1])

    outputs = tf.convert_to_tensor(outputs)  # (len_out,batch_size,1,1)
    outputs = tf.squeeze(outputs, -1)  # (len_out,batch_size,1)
    outputs = tf.transpose(outputs, perm=[1, 0, 2])  # (batch_size,len_out,1)
    model = tf.keras.Model([input_shape_exo, input_shape_target], outputs)
    return model

from datetime import datetime, time, timedelta
def strTotime(date_str, time_str):
  # 解析日期部分
  date = datetime.strptime(date_str, '%d/%m/%Y').date()

  # 解析时间部分
  time = datetime.strptime(time_str, '%H:%M').time()

  # 将日期和时间合并成 datetime 对象
  combined_datetime = datetime.combine(date, time)
  return combined_datetime

def predict():
    nb_filters = 96
    dim_filters = 2
    dilation_depth = 4  # len_seq = 32
    use_bias = False
    res_l2 = 0
    final_l2 = 0
    len_seq = 32
    batch_size = 128
    len_out = 5
    dim_exo = 7
    model = model_wavenet_timeseries(len_seq, len_out, dim_exo, nb_filters, dim_filters, dilation_depth, use_bias, res_l2, final_l2,batch_size=1)
    model.load_weights("./weights_train.hdf5")
    with open('scalerTem.pkl', 'rb') as f:
        scalerTem = pickle.load(f)
    with open('scalerOther.pkl', 'rb') as f:
        scalerOther = pickle.load(f)
    current_time = datetime.now().replace(second=0, microsecond=0)
    #仅供测试使用
   # current_time = datetime(2012, 3, 13, 19, 45, 0)
    predictData = pd.DataFrame(columns=['time', 'temperature'])
    # 调用数据库处理数据并预测  8小时之前的数据  比如当前19.30  要采集11.31到19.30之间的数据，间隔一分钟 总共480个数据 分15组，每组32个 15分钟为一跳
    data = pd.DataFrame(columns=['time','Relative_humidity_room','Meteo_Rain','Meteo_Wind','Meteo_Sun_light_in_west_facade','Meteo_Sun_light_in_east_facade','Meteo_Sun_irradiance','Outdoor_relative_humidity_Sensor','Indoor_temperature_room'])
    data = pd.read_csv('newdata.csv')
    data.drop(['dateTime'],axis=1,inplace=True)
    data = data.astype(dtype="float32")
    grouped_dfs = []

    # 按照要求分组
    for i in range(15):
        start_index = i  # 每组的起始索引
        indices = list(range(start_index, 480, 15))  # 计算每组的索引
        grouped_df = data.iloc[indices]  # 按照索引取出数据
        grouped_dfs.append(grouped_df)
    for i in range(15):
        time = current_time - timedelta(minutes=i)
        #调用数据库处理数据并预测
        data_i = tf.convert_to_tensor(grouped_dfs[i])
        x_train_other = scalerOther.transform(data_i[:, :7])
        x_train_tem = scalerTem.transform(tf.reshape(data_i[:, 7], shape=(-1, 1)))
        pred = model.predict((np.expand_dims(x_train_other, axis=0), np.expand_dims(x_train_tem, axis=0)), verbose=1, batch_size=1)
        pred = scalerTem.inverse_transform(np.reshape(pred, newshape=(5, 1)))
        #一个小时15分钟的温度预测
        for j in range(5):
            new_time = time + timedelta(minutes = (j + 1) * 15)
            new_temperature = pred[j]
            new_row = pd.DataFrame({'time': [new_time], 'temperature': [new_temperature]})
            predictData = pd.concat([predictData, new_row]).sort_values(by='time').reset_index(drop=True)
    return predictData

if __name__ == '__main__':
    predictData = predict()
    print(predictData)
