import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import dataset, read_target, get_data
from multiprocessing import Pool
import os

df = pd.read_csv('result.csv', index_col=0)
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)
print(df.mean())

# df = pd.read_csv('./data/code.csv', index_col=0)
# codes = []
# for i in range(df.shape[0]):
#     codes.append('%06d'% df['code'][i:i+1].values[0])

# date = np.unique(pd.read_csv('./data/target.csv').date.values)[361:]

# data = pd.DataFrame()
# for d in date:
#     df = dataset(d, codes)
#     data = data.append(df)
# print(data.shape)
# data.to_csv('./data/raw_data/data_12.csv')

# def read_data(i, days):
#     print('Start process %d to read data.' % os.getpid())
#     data = pd.DataFrame()
#     for d in days:
#         df = dataset(d, codes)
#         data = data.append(df)
#     print(data.shape)
#     data.to_csv('./data/raw_data/data_%02d.csv' % i)

# p = Pool(12)
# for i in range(12):
#     p.apply_async(read_data, args=(i, date[i*31:(i+1)*31]))
# p.close()
# p.join()
# print('Finished!')

# data = pd.DataFrame()
# for i in range(6, 13):
#     df = pd.read_csv('./data/raw_data/data_%02d.csv' % i, index_col=0)
#     data = data.append(df)
# print(data.shape)
# data.to_csv('./data/data.csv')
