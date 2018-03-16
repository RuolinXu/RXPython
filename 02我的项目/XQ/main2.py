import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from XQ.StockData2 import StockData2
from sklearn import model_selection, metrics
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def get_symbol_kl(symbol, start='2000-01-01', end='2100-01-01', ktype='K_DAY'):
    kline_pd = StockData2.kline_pd_from_db(symbol, start=start, end=end, ktype=ktype)
    return kline_pd


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


def gen_data_set(symbol, clsFunc=lambda x: np.where(x > 0, 1, 0), xList=[1], yIndex=0,
                 start='2000-01-01', end='2100-01-01', ktype='K_DAY', show=False):
    """
    生成机器学习使用的数据
    :param symbol: "US.BABA"
    :param clsFunc: 分类的方法
    :param xList: 特征筛选[1,3,4] 表示选择 特征 1 3 4 作为 x值
    :param yIndex: y 值的取值列
    :param start:
    :param end:
    :param ktype:
    :param show:
    :return:
    """
    symbol_kl = get_symbol_kl(symbol, start=start, end=end, ktype=ktype)
    symbol_kl_feature = gen_symbol_kl_feature(symbol_kl)

    # Dataframe -> matrix
    feature_np = symbol_kl_feature.as_matrix()
    # x特征矩阵
    data_x = feature_np[:, xList] if isinstance(xList, list) else feature_np[:, 1:]
    # 回归训练的连续值y
    data_y = feature_np[:, yIndex]
    # 分类训练的离散值y，之后分类技术使用
    data_y_classification = clsFunc(data_y)  # np.where(train_y_regress > 0, 1, 0)

    if show:
        print('feature.shape:', symbol_kl_feature.shape)
        print('feature.head():\n', symbol_kl_feature.head())
        print('data_x[:5]\n{}\ndata_y[:5]\n{}\ndata_y_classification[:5]:\n{}\n'.format(data_x[:5], data_y[:5],
              data_y_classification[:5]))

    return data_x, data_y, data_y_classification, symbol_kl_feature


def regress_process(estimator, train_x, train_y_regress, test_x, test_y_regress):
    # 训练训练集数据
    estimator.fit(train_x, train_y_regress)
    # 使用训练好的模型预测测试集对应的y，即根据usFB的走势特征预测股价涨跌幅度
    predictions = estimator.predict(test_x)

    # 绘制实际股价涨跌幅度  蓝
    plt.plot(test_y_regress.cumsum())
    # 绘制通过模型预测的股价涨跌幅度   绿
    plt.plot(predictions.cumsum())

    # 针对训练集数据做交叉验证
    scores = model_selection.cross_val_score(estimator, train_x, train_y_regress,
                                             cv=10, scoring='mean_squared_error')
    # mse开方 -> rmse
    print("{}===TrainingSet Cross Validation RMSE:{:.2f}"
          .format(estimator.__class__.__name__, -np.mean(np.sqrt(-scores))))


def classification_process(estimator, train_x, train_y_classification, test_x, test_y_classification):

    # 训练数据，这里分类要所以要使用y_classification
    estimator.fit(train_x, train_y_classification)
    # 使用训练好的分类模型预测测试集对应的y，即根据usFB的走势特征预测涨跌
    predictions = estimator.predict(test_x)

    # 针对训练集数据做交叉验证scoring='accuracy'，cv＝10
    scores = model_selection.cross_val_score(estimator, train_x, train_y_classification,
                                             cv=10, scoring='accuracy')
    # 所有交叉验证的分数取平均值
    print("{}\nTrainingSet Cross Validation accuracy mean:{:.2f}"
          .format(estimator.__class__.__name__, np.mean(scores)))
    print("TestSet prediction:")
    # 度量准确率
    print("accuracy = {:.2f}".format(metrics.accuracy_score(test_y_classification, predictions)))
    # 度量查准率
    print("precision_score = {:.2f}".format(metrics.precision_score(test_y_classification, predictions)))
    # 度量回收率
    print("recall_score = {:.2f}" % (metrics.recall_score(test_y_classification, predictions)))
    confusion_matrix = metrics.confusion_matrix(test_y_classification, predictions)
    print("Confusion Matrix: \n", confusion_matrix)
    # # print("          Predicted")
    # # print("         |  0  |  1  |")
    # # print("         |-----|-----|")
    # # print("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
    # #                                   confusion_matrix[0, 1]))
    # # print("Actual   |-----|-----|")
    # # print("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
    # #                                   confusion_matrix[1, 1]))
    # # print("         |-----|-----|")
    print("metrics.classification_report:")
    print(metrics.classification_report(test_y_classification, predictions))


def split_xy_train_test(x, y, test_size=0.5, random_state=0, show=False):
    # 通过train_test_split将原始训练集随机切割为新训练集与测试集
    train_x, test_x, train_y, test_y = \
        model_selection.train_test_split(x, y, test_size=test_size, random_state=random_state)

    if show:
        print("xShape:{}, yShape{}".format(x.shape, y.shape))
        print("train_xShape:{},train_yShape{}".format(train_x.shape, train_y.shape))
        print("test_xShape: {}, test_yShape: {}".format(test_x.shape, test_y.shape))

    return train_x, test_x, train_y, test_y


def check_features_importance(estimator, features, train_x, train_y_classification):
    def importances_coef_pd(estimator, x, y):
        # 训练数据模型
        estimator.fit(x, y)
        if hasattr(estimator, 'feature_importances_'):
            return pd.DataFrame(
                {'feature': list(features.columns[1:]),
                 'importance': estimator.feature_importances_}).sort_values('importance')

        elif hasattr(estimator, 'coef_'):
            return pd.DataFrame(
                {"columns": list(features.columns)[1:], "coef": list(estimator.coef_.T)}).sort_values('coef')
        else:
            print('estimator not hasattr feature_importances_ or coef_!')

    # 对训练后的模型特征的重要度进行判定，重要程度由小到大，表10-4所示
    print('importances_coef_pd(estimator):\n', importances_coef_pd(estimator, train_x, train_y_classification))

    def feature_selection(estimator, x, y):
        """
            支持度评级
        """
        from sklearn.feature_selection import RFE
        selector = RFE(estimator)
        selector.fit(x, y)
        print('RFE selection')
        print(pd.DataFrame(
            {'support': selector.support_, 'ranking': selector.ranking_},
            index=features.columns[1:]))

    print('\nfeature_selection(estimator, train_x, train_y_classification):\n')
    feature_selection(estimator, train_x, train_y_classification)

# def decision_tree_classification_process(features, train_x, train_y_classification):
#     from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
#     from sklearn import tree
#     import os
#
#     estimator = DecisionTreeRegressor(max_depth=2, random_state=1)
#
#     def graphviz_tree(estimator, features, x, y):
#         if not hasattr(estimator, 'tree_'):
#             print('only tree can graphviz!')
#             return
#         estimator.fit(x, y)
#         tree.export_graphviz(estimator.tree_, out_file='graphviz.dot', feature_names=features)
#         os.system('dot -T png graphviz.dot -o graphviz.png')
#
#     graphviz_tree(estimator, features.columns[1:], train_x, train_y_classification)


def regress_process_test():
    data_x, data_y, data_y_classification, symbol_kl_feature = gen_data_set('US.BABA')

    # 实例化线性回归对象estimator
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()

    # pipeline套上 degree=3 ＋ LinearRegression
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.linear_model import LinearRegression
    # estimator = make_pipeline(PolynomialFeatures(degree=2),
    #                           LinearRegression())

    # AdaBoost
    # from sklearn.ensemble import AdaBoostRegressor
    # estimator = AdaBoostRegressor(n_estimators=100)

    # RandomForest
    # from sklearn.ensemble import RandomForestRegressor
    # estimator = RandomForestRegressor(n_estimators=100)

    train_x, test_x, train_y, test_y = split_xy_train_test(data_x, data_y, show=True)
    # 将回归模型对象，训练集x，训练集连续y值，测试集x，测试集连续y传入
    regress_process(estimator, train_x, train_y, test_x, test_y)
    plt.show()


def classification_process_test_1():
    data_x, data_y, data_y_classification, symbol_kl_feature = gen_data_set('US.BABA', xList="All")

    # from sklearn.linear_model import LogisticRegression
    # estimator = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

    # from sklearn.svm import SVC
    # estimator = SVC(kernel='rbf')

    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100)

    # train_x, test_x, train_y, test_y = split_xy_train_test(data_x, data_y_classification)

    # classification_process(estimator, train_x, train_y, test_x, test_y)

    check_features_importance(estimator, symbol_kl_feature, data_x, data_y_classification)


if __name__ == '__main__':
    # regress_process_test()
    classification_process_test_1()

