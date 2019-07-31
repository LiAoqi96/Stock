import pandas as pd
import numpy as np
from load_data import get_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 5, 7, 8"

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

with tf.device('/cpu:0'):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(735, 10)))
    model.add(LSTM(100))
    model.add(Dense(1))

early_stopping = EarlyStopping('loss', 0.0001, 5)
model = multi_gpu_model(model, 4)
model.compile(loss='mse', optimizer=Adam(1e-4))

result = []
predict = pd.DataFrame()
for i in range(0, 82, 3):
    x_train, y_train, x_test, y_test = get_data(i)

    model.fit(x_train, y_train, batch_size=2048, epochs=100, callbacks=[early_stopping])

    y_pred = model.predict(x_test, batch_size=500)
    r = pd.DataFrame({'change': y_test.flatten(), 'pred': y_pred.flatten()})

    for j in range(3):
        df = pd.DataFrame({'predict': y_pred[j::3].flatten()})
        predict = predict.append(df)
        result.append(r[j::3].corr().values[0, 1])

    df = pd.DataFrame({'p': np.array(result)})
    df.to_csv('result.csv')

predict.to_csv('./predict.csv')
model.save('./model/model.h5')