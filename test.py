import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import LSTM_Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

model = LSTM_Model(gpus=4)

for i in range(0, model.rolling_count, 3):
    model.rolling(i)

model.save_model()
