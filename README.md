第一部分 数据处理

原始数据共有51个特征，每天记录4802个时间点的数据，我们将数据进行以下的处理：
ask1-10:一到十档盘口卖价取平均值，代替原有十档数据
bid1-10:一到十档盘口买价取平均值，代替原有十档数据
ask_vol1-10:一到十档盘口卖量取平均值，代替原有十档数据
bid_vol1-10:一到十档盘口买量取平均值，代替原有十档数据
amount: 	tick内的累计交易量，舍弃
last: 		最新成交价，取指数滑动平均值代替
avebid: 	平均委托买入价，舍弃
aveoff: 	平均委托卖出价，舍弃
trade: 	累计成交笔数，舍弃
totbid: 	截至当前未成交的委托买入量
totoff: 	截至当前未成交的委托卖出量
vol: 		累计成交量，取差分
open:	开盘价，舍弃
low:		最低价	，取指数滑动平均值代替
High: 	最高价，取指数滑动平均值代替
对每天的4802个时刻数据进行采样，间隔为100，每天保留的时刻为49个
，经过特征筛选处理后，保留下10个特征，每天的数据变为49x10。

第二部分 网络模型

采用双层LSTM加全连接层的结构，模型如下：

第三部分模型训练

选取前15天的数据作为输入，输入数据采用最大最小值归一化，每天收益率减去中证500收益率作为目标，收益率除以20加上0.5，训练收益率clip（0，1），优化函数选择Adam，学习率0.0001，batch_size为2048。Loss选用mse，训练采用early_stopping，当loss下降小于0.0001后的5个epoch停止训练

第四部分 实验结果

每次rolling后保留网络参数，用到下一次训练中去，进行了28次rolling，其中后18次为5-6到7-18号的结果，相关系数平均值为0.085.
![结果图片](https://github.com/LiAoqi96/Stock/edit/master/p.png)
