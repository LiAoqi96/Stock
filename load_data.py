import os
import pandas as pd
import numpy as np
import load_md as foo
import time

def dataset(date, codes=['000002']):

    keys = ['totbid', 'totoff', 'vol', 'last', 'low', 'high']
    keys.extend(['bid' + str(x) for x in range(1, 11)])
    keys.extend(['ask' + str(x) for x in range(1, 11)])
    keys.extend(['bid_vol' + str(x) for x in range(1, 11)])
    keys.extend(['ask_vol' + str(x) for x in range(1, 11)])

    data = pd.DataFrame()

    df = read_data(date, codes, keys)
    t1 = time.time()
    for c in codes:
        temp = pd.DataFrame(df[c].values)
        temp.columns = keys
        temp['code'] = c

        if temp.isnull().sum().max() >= 4802:
            continue

        temp['vol'] = temp['vol'].diff().fillna(0)
        temp['vol'][temp['vol'] < 0] = 0

        temp['ask'] = temp[['ask' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp['bid'] = temp[['bid' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp.drop(['ask' + str(x) for x in range(1, 11)], axis=1, inplace=True)
        temp.drop(['bid' + str(x) for x in range(1, 11)], axis=1, inplace=True)

        temp['ask_vol'] = temp[['ask_vol' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp['bid_vol'] = temp[['bid_vol' + str(x) for x in range(1, 11)]].mean(axis=1)
        temp.drop(['ask_vol' + str(x) for x in range(1, 11)], axis=1, inplace=True)
        temp.drop(['bid_vol' + str(x) for x in range(1, 11)], axis=1, inplace=True)

        m_lst = ['last', 'bid', 'ask']
        for i in m_lst:
            temp[i+'_mean'] = temp[i].ewm(span=4).mean()
            temp.drop(i, axis=1, inplace=True)

        data = data.append(temp[::100])
    print('Finished %d data processing, cost %.3fs' % (date, time.time() - t1))
    return data

def read_data(date, codes=['000002'], keys=None):
    fp64_keys = keys[:3]
    float_keys = keys[3:]

    t1 = time.time()
    data = pd.DataFrame()
    for i in fp64_keys:
        df = foo.get_mem_data_by_tick(date, i, codes=codes, dtype='float64').astype('float32')
        data = pd.concat([data, df], axis=1)

    for i in float_keys:
        df = foo.get_mem_data_by_tick(date, i, codes=codes, dtype='float32')
        data = pd.concat([data, df], axis=1)

    print('Finished %d, cost %.3fs' % (date, time.time() - t1))
    return data

def get_data(d=0):
    data = pd.read_csv('./data/data.csv', index_col=0).groupby('code')
    target = pd.read_csv('./data/target.csv').groupby('code')

    codes = pd.read_csv('./data/code.csv', index_col=0).code.values
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for c in codes:
        try:
	        df = data.get_group(c)
	        obj = target.get_group(c)[186:]
        except KeyError:
            continue

        df.drop('code', axis=1, inplace=True)

        if df.shape[0] < 49*189:
            continue

        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        for j in range(d+15, d+109):
            if j < d+105:
                x_train.append(df[(j-15)*49:j*49].values)
                y_train.append(obj['change'][j:j+1].values)
            elif j == d+105:
                pass
            else:
                x_test.append(df[(j-15)*49:j*49].values)
                y_test.append(obj['change'][j:j+1].values)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = y_train / 20 + 0.5
    y_train = np.clip(y_train, 0, 1)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = y_test / 20 + 0.5
    
    return x_train, y_train, x_test, y_test

def read_target(year, month=1):
    df = pd.read_csv('./data/raw_data/target.csv', index_col=0) # /home/sharedFold/fwd_return/return_1d.csv
    df.columns = ['date', 'code', 'change']

    target = pd.DataFrame()
    for day in range(19, 20):
        target = target.append(df[df.date == year*10000+month*100+day])

    return target

if __name__ == '__main__':
    codes = list(pd.read_csv('./data/code.csv', index_col=0).values)
    print(len(codes))

    for i in (7, 8):
        for j in range(1, 19):
            if not os.path.exists(os.path.join('/data/stock/newSystemData/rawdata/universe/TOP2000', '2019%02d%02d' % (i, j))):
                continue

            with open(os.path.join('/data/stock/newSystemData/rawdata/universe/TOP2000', '20190102')) as f:
                for c in f:
                    codes.append(c.strip())

    codes = sorted(codes)
    df = pd.DataFrame({'code': codes})
    print(df.shape)
    df.to_csv('./data/code.csv')
