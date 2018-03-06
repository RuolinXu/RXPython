import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from XQ.StockData2 import StockData2


def gen_symbol_kl(symbol, start='2000-01-01', end='2100-01-01'):
    kline_pd = StockData2.kline_pd_from_db(symbol, start=start, end=end, ktype='K_DAY')
    return kline_pd


def gen_stockdata_dict():
    choice_symbols = ['US.BABA', 'US.NVDA']

    stockdata_dict = {}
    for symbol in choice_symbols:
        stockdata_dict[symbol] = gen_symbol_kl(symbol)
    return stockdata_dict


def gen_symbol_kl_feature(symbol_kl):
    # y值使用close.pct_change即涨跌幅度
    symbol_kl['regress_y'] = symbol_kl.Close.pct_change()
    # 前天收盘价格
    symbol_kl['bf_yesterday_close'] = 0
    # 昨天收盘价格
    symbol_kl['yesterday_close'] = 0
    # 昨天收盘成交量
    symbol_kl['yesterday_volume'] = 0
    # 前天收盘成交量
    symbol_kl['bf_yesterday_volume'] = 0

    # 对齐特征，前天收盘价格即与今天的收盘错2个时间单位，[2:] = [:-2]
    symbol_kl['bf_yesterday_close'][2:] = \
        symbol_kl['Close'][:-2]
    # 对齐特征，前天成交量
    symbol_kl['bf_yesterday_volume'][2:] = \
        symbol_kl['Volume'][:-2]
    # 对齐特征，昨天收盘价与今天的收盘错1个时间单位，[1:] = [:-1]
    symbol_kl['yesterday_close'][1:] = \
        symbol_kl['Close'][:-1]
    # 对齐特征，昨天成交量
    symbol_kl['yesterday_volume'][1:] = \
        symbol_kl['Volume'][:-1]

    # 特征1: 价格差
    symbol_kl['feature_price_change'] = \
        symbol_kl['yesterday_close'] - \
        symbol_kl['bf_yesterday_close']

    # 特征2: 成交量差
    symbol_kl['feature_volume_Change'] = \
        symbol_kl['yesterday_volume'] - \
        symbol_kl['bf_yesterday_volume']

    # 特征3: 涨跌sign
    symbol_kl['feature_sign'] = np.sign(
        symbol_kl['feature_price_change'] * symbol_kl['feature_volume_Change'])

    # 特征4: 周几
    # symbol_kl['feature_date_week'] = symbol_kl['date_week']

    """
        构建噪音特征, 因为猪老三也不可能全部分析正确真实的特征因素
        这里引入一些噪音特征
    """
    # 成交量乘积
    symbol_kl['feature_volume_noise'] = \
        symbol_kl['yesterday_volume'] * \
        symbol_kl['bf_yesterday_volume']

    # 价格乘积
    symbol_kl['feature_price_noise'] = \
        symbol_kl['yesterday_close'] * \
        symbol_kl['bf_yesterday_close']

    # 将数据标准化
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()
    symbol_kl['feature_price_change'] = scaler.fit_transform(
        symbol_kl['feature_price_change'].values.reshape(-1, 1))
    symbol_kl['feature_volume_Change'] = scaler.fit_transform(
        symbol_kl['feature_volume_Change'].values.reshape(-1, 1))
    symbol_kl['feature_volume_noise'] = scaler.fit_transform(
        symbol_kl['feature_volume_noise'].values.reshape(-1, 1))
    symbol_kl['feature_price_noise'] = scaler.fit_transform(
        symbol_kl['feature_price_noise'].values.reshape(-1, 1))

    # 只筛选feature_开头的特征和regress_y，抛弃前两天数据，即[2:]
    symbol_kl_feature = symbol_kl.filter(
        regex='regress_y|feature_*')[2:]
    return symbol_kl_feature


def gen_train_sets(show=False):
    stockdata_dict = gen_stockdata_dict()
    stockdata_feature = None
    for symbol in stockdata_dict:
        symbol_kl = stockdata_dict[symbol]
        symbol_kl_feature = gen_symbol_kl_feature(symbol_kl)
        stockdata_feature = symbol_kl_feature if stockdata_feature is None \
            else stockdata_feature.append(symbol_kl_feature)

    # Dataframe -> matrix
    feature_np = stockdata_feature.as_matrix()
    # x特征矩阵
    train_x = feature_np[:, 1:]
    # 回归训练的连续值y
    train_y_regress = feature_np[:, 0]
    # 分类训练的离散值y，之后分类技术使用
    # noinspection PyTypeChecker
    train_y_classification = np.where(train_y_regress > 0, 1, 0)

    if show:
        print('pig_three_feature.shape:', stockdata_feature.shape)
        print('pig_three_feature.tail():\n', stockdata_feature.tail())
        print('train_x[:5], train_y_regress[:5], train_y_classification[:5]:\n', train_x[:5], train_y_regress[:5],
              train_y_classification[:5])

    return train_x, train_y_regress, train_y_classification, stockdata_feature


def gen_test_sets(symbol, start='2017-01-01', end='2100-01-01'):
    # 真实世界走势数据转换到老三的世界
    symbol_kl = gen_symbol_kl(symbol, start, end)
    # 由走势转换为特征dataframe通过gen_pig_three_feature
    symbol_kl_feature_test = gen_symbol_kl_feature(symbol_kl)
    # 转换为matrix
    feature_np_test = symbol_kl_feature_test.as_matrix()
    # 从matrix抽取y回归
    test_y_regress = feature_np_test[:, 0]
    # y回归 －> y分类
    # noinspection PyTypeChecker
    test_y_classification = np.where(test_y_regress > 0, 1, 0)
    # 从matrix抽取x特征矩阵
    test_x = feature_np_test[:, 1:]
    return test_x, test_y_regress, test_y_classification, symbol_kl_feature_test


def regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress):
    # 训练训练集数据
    estimator.fit(train_x, train_y_regress)
    # 使用训练好的模型预测测试集对应的y，即根据usFB的走势特征预测股价涨跌幅度
    test_y_prdict_regress = estimator.predict(test_x)

    # 绘制usFB实际股价涨跌幅度
    plt.plot(test_y_regress.cumsum())
    # 绘制通过模型预测的usFB股价涨跌幅度
    plt.plot(test_y_prdict_regress.cumsum())

    # 针对训练集数据做交叉验证
    from sklearn import cross_validation
    scores = cross_validation.cross_val_score(estimator, train_x,
                                              train_y_regress, cv=10,
                                              scoring='mean_squared_error')
    # mse开方 -> rmse
    mean_sc = -np.mean(np.sqrt(-scores))
    print('{} RMSE: {}'.format(estimator.__class__.__name__, mean_sc))


def foo():
    train_x, train_y_regress, train_y_classification, stockdata_feature = gen_train_sets()
    test_x, test_y_regress, test_y_classification, symbol_kl_feature_test = gen_test_sets('US.BABA')
    # 实例化线性回归对象estimator
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    # 将回归模型对象，训练集x，训练集连续y值，测试集x，测试集连续y传入
    regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress)
    plt.show()


if __name__ == '__main__':
    foo()

