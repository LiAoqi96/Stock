from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
import tensorflow as tf
from load_data import get_data
import pandas as pd
import numpy as np


class LSTM_Model:
    def __init__(self, gpus=1):
        with tf.device('/cpu:0'):
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(735, 10)))
            model.add(LSTM(100))
            model.add(Dense(1))

        if gpus > 1:
            self.model = multi_gpu_model(model, gpus)
        else:
            self.model = model
        self.model.compile(loss='mse', optimizer=Adam(1e-4))
        self.result = []
        self.predict = pd.DataFrame()
    
    def rolling(self, i=0):
        x_train, y_train, x_test, y_test = get_data(i)
        early_stopping = EarlyStopping('loss', 0.0001, 5)
        self.model.fit(x_train, y_train, batch_size=2048, epochs=100, callbacks=[early_stopping])

        y_pred = self.model.predict(x_test, batch_size=500)
        r = pd.DataFrame({'change': y_test.flatten(), 'pred': y_pred.flatten()})

        for j in range(3):
            df = pd.DataFrame({'predict': y_pred[j::3].flatten()})
            self.predict = self.predict.append(df)
            self.result.append(r[j::3].corr().values[0, 1])

        df = pd.DataFrame({'p': np.array(self.result)})
        df.to_csv('result.csv')

    def finish(self):
        self.predict.to_csv('./predict.csv')
        self.model.save('./model/model.h5')
        
        # df = pd.read_csv('./predict.csv', index_col=0)
        # code = pd.read_csv('./data/r_code.csv', index_col=0).code.values

        # date = np.unique(pd.read_csv('./data/target.csv').date.values)[106:]
        # for i in range(77):
        #     result = df[i*1346:(i+1)*1346]
        #     result.loc[:, 'code'] = code
        #     result.loc[:, 'code'] = result['code'].apply(lambda x: str(x).zfill(6))
        #     result.set_index('code', inplace=True)
        #     result = result.sort_values('predict', ascending=False)
        #     result.to_csv('./data/csv/%d.csv' % date[i], header=None)