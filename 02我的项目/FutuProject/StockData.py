from collections import namedtuple
from collections import OrderedDict
from SQLite3DB import SQLite3DB, DataCondition
from functools import reduce
import numpy as np
import datetime
from futuquant.open_context import *
import matplotlib.pyplot as plt
import matplotlib.finance as mpf


DB_PATH = r'./StockDB.db'


class StockData(object):
    """history data from DB"""
    def __init__(self, stockcode, fromdate='2000-01-01', todate='2100-01-01'):
        self.stockcode = stockcode
        self.fromdate = fromdate
        self.todate = todate
        self.__sorted_rs = self.__load_sorted_rs()
        self.__time_array = [x[6] for x in self.__sorted_rs]
        self.days_dict = self.__init_days_dict()
        self.__sorted_rs = None  # free some memory

    def __load_sorted_rs(self):
        db = SQLite3DB(DB_PATH)
        db.open()
        rs = db.table("KLine1M") \
            .select("Open", "Low", "High", "Close", "Volume", "Turnover", "KLTime") \
            .where(DataCondition(("=", "AND"), StockCode=self.stockcode),
                   DataCondition((">", "AND"), KLTime=self.fromdate),
                   DataCondition(("<", "AND"), KLTime=self.todate)) \
            .fetchall()
        # 0=Open 1=Low 2=High 3=Close 4=Volume 5=Turnover 6=KLTime
        db.close()
        sorted_rs = sorted(rs, key=lambda a: a[6])  # sorted by time
        return sorted_rs

    def __init_days_dict(self):
        stock_nametuple = namedtuple(self.stockcode.split('.')[1],
                                     ('Open','Low','High','Close','Volume','Turnover','KLTime'))
        # 0=Open 1=Low 2=High 3=Close 4=Volume 5=Turnover 6=KLTime
        # days_dict = OrderedDict((x6, stock_nametuple(x0,x1,x2,x3,x4,x5,x6))
        #                         for x0,x1,x2,x3,x4,x5,x6 in
        #                         zip([x[0] for x in self.__sorted_rs],
        #                             [x[1] for x in self.__sorted_rs],
        #                             [x[2] for x in self.__sorted_rs],
        #                             [x[3] for x in self.__sorted_rs],
        #                             [x[4] for x in self.__sorted_rs],
        #                             [x[5] for x in self.__sorted_rs],
        #                             [x[6] for x in self.__sorted_rs]))
        days_dict = OrderedDict((x[6], stock_nametuple(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
                                for x in self.__sorted_rs)

        return days_dict

    def __iter__(self):
        for key in self.days_dict:
            yield self.days_dict[key]

    def __getitem__(self, ind):
        time_key = self.__time_array[ind]
        return self.days_dict[time_key]

    def __str__(self):
        start = self.__time_array[1]
        end = self.__time_array[-1]
        return str(self.days_dict[start]) + '\n...\n' + str(self.days_dict[end])

    @staticmethod
    def add_mins_to_time(current, ind, period=390, shour=9, smin=30):
        current_time = datetime.strptime(current, '%Y-%m-%d %H:%M:%S')
        start_time = datetime(current_time.year, current_time.month, current_time.day, shour, smin)
        delta = int((current_time - start_time).seconds / 60)
        days = int((ind + delta) / period)
        mins = int((ind + delta) % period)
        if ind < 0 and abs(ind) > delta:
            days -= 1
        time_tar = start_time + timedelta(days=days, minutes=mins)
        return time_tar.strftime('%Y-%m-%d %H:%M:%S')

    @classmethod
    def __get_change_array(cls, tar, base):
        pp_array = [(p1, p2) for p1, p2 in zip(base[:-1], tar[1:])]
        change_array = list(map(lambda pp: reduce(lambda a, b: (b - a) / a, pp), pp_array))
        change_array.insert(0, 0)
        return change_array

    @classmethod
    def __get_interval_array(cls, tar, base):
        pp_array = [(p1, p2) for p1, p2 in zip(base[:-1], tar[1:])]
        interval_array = list(map(lambda pp: reduce(lambda a, b: b - a, pp), pp_array))
        interval_array.insert(0, 0)
        return interval_array

    def get_kp_array(self, fromdate='2000-01-01', todate='2100-01-01', rate=0.03):
        """获取时间段中的关键点
        :param fromdate:
        :param todate:
        :param rate:
        :return: namedtuple_list('Ind', 'Price', 'KLTime', 'D')
        """
        p_nametuple = namedtuple(self.stockcode.split('.')[1], ('Ind', 'Price', 'KLTime', 'D'))
        kp_array = []
        for key in self.days_dict:
            if key > todate:
                break
            if fromdate < key < todate:
                _ind = self.__time_array.index(key)
                _open = self.days_dict[key].Open
                _low = self.days_dict[key].Low
                _high = self.days_dict[key].High
                _ktime = self.days_dict[key].KLTime
                if len(kp_array) == 0:
                    kp_array.append(p_nametuple(_ind, _open, _ktime, 0))
                    continue
                if kp_array[-1].D == 0:
                    if _high > (kp_array[-1].Price * (1+rate)):
                        kp_array.append(p_nametuple(_ind, _high, _ktime, 1))
                        continue
                    if _low < (kp_array[-1].Price * (1-rate)):
                        kp_array.append(p_nametuple(_ind, _low, _ktime, -1))
                        continue
                if kp_array[-1].D == 1:
                    if _high > kp_array[-1].Price:
                        kp_array[-1] = p_nametuple(_ind, _high, _ktime, 1)
                        continue
                    elif _low < (kp_array[-1].Price * (1-rate)):
                        kp_array.append(p_nametuple(_ind, _low, _ktime, -1))
                        continue
                if kp_array[-1].D == -1:
                    if _low < kp_array[-1].Price:
                        kp_array[-1] = p_nametuple(_ind, _low, _ktime, -1)
                        continue
                    elif _high > (kp_array[-1].Price * (1 + rate)):
                        kp_array.append(p_nametuple(_ind, _high, _ktime, 1))
                        continue
        return kp_array

    def get_keypoint_report(self, kp_array):
        price_array = np.array([x.Price for x in kp_array])
        ind_array = np.array([x.Ind for x in kp_array])
        change_array = self.__get_change_array(price_array, price_array)
        interval_array = self.__get_interval_array(ind_array, ind_array)
        if kp_array[1].D == 1:      # 如果第一个节点是涨
            up_change_array = np.array(change_array[1::2])
            up_interval_array = np.array(interval_array[1::2])
            down_change_array = np.array(change_array[2::2])
            down_interval_array = np.array(interval_array[2::2])
        if kp_array[1].D == -1:     # 如果第一个节点是跌
            up_change_array = np.array(change_array[2::2])
            up_interval_array = np.array(interval_array[2::2])
            down_change_array = np.array(change_array[1::2])
            down_interval_array = np.array(interval_array[1::2])
        print("平均涨幅%f,最大涨幅%f,最小涨幅%f, 平均周期%f分钟" %
              (up_change_array.mean(), up_change_array.max(), up_change_array.min(), up_interval_array.mean()))
        print("平均跌幅%f,最大跌幅%f,最小跌幅%f, 平均周期%f分钟" %
              (down_change_array.mean(), down_change_array.min(), down_change_array.max(), down_interval_array.mean()))
        print(change_array)

        if kp_array[-1].D == 1:  # 如果最后一个节点是涨 显示下一节点的目标价
            price_tar = kp_array[-1].Price * (1 + down_change_array.mean())
            time_delta = down_interval_array.mean()
        if kp_array[-1].D == -1:
            price_tar = kp_array[-1].Price * (1 + up_change_array.mean())
            time_delta = up_interval_array.mean()

        time_str = self.add_mins_to_time(kp_array[-1].KLTime, time_delta)
        print("目标价%f,时间点%s" % (price_tar, time_str))
        plt.figure(figsize=(16, 9))
        plt.plot(ind_array, price_array)
        plt.show()

    # @staticmethod
    # def get_data_view():
    #     # price_array = np.array([x.Price for x in kp_array])
    #     # ind_array = np.array([x.Ind for x in kp_array])
    #     # np.save(r'.\\price_array', price_array)
    #     # np.save(r'.\\ind_array', ind_array)
    #
    #     price_array = np.load('.\\price_array.npy')
    #     ind_array = np.load('.\\ind_array.npy')
    #     change_array = StockTradeDays.__get_change_array(price_array, price_array)
    #     interval_array = StockTradeDays.__get_interval_array(ind_array, ind_array)
    #
    #     if change_array[1]>0:   #
    #         up_change_array = np.array(change_array[1::2])
    #         up_interval_array = np.array(interval_array[1::2])
    #         down_change_array = np.array(change_array[2::2])
    #         down_interval_array = np.array(interval_array[2::2])
    #     if change_array[1]<0:
    #         up_change_array = np.array(change_array[2::2])
    #         up_interval_array = np.array(interval_array[2::2])
    #         down_change_array = np.array(change_array[1::2])
    #         down_interval_array = np.array(interval_array[1::2])
    #
    #     # print(np_kp)
    #     # x = [a.Ind for a in np_kp]
    #     # y = [a.Price for a in np_kp]
    #     # x = np.linspace(up_change_array.min(), up_change_array.max(), 10)
    #     print(up_interval_array.mean())
    #     print(up_interval_array.std())
    #     print(up_interval_array.max())
    #     print(up_interval_array.min())
    #     plt.hist(down_interval_array, bins=100)
    #     # plt.plot(down_change_array)
    #     plt.show()

    def update_db(self):
        """调用富途接口，把数据添加到数据库 """
        quote_context = OpenQuoteContext(host='127.0.0.1', async_port=11111)
        market = self.stockcode.split('.')[0]
        from_date = self.__time_array[-1][0:10]
        to_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        ret_code, content = quote_context.get_trading_days(market, from_date, to_date)
        if ret_code != RET_OK:
            print("RTDataTest: error, msg: %s" % content)
            return RET_ERROR, content
        trade_days = sorted(content)           # 排序交易日
        trade_day_range = trade_days[::3]      # 每次取3天
        if trade_day_range[-1] != trade_days[-1]:
            trade_day_range.append(trade_days[-1])
        db = SQLite3DB(DB_PATH)
        db.open()
        count = 0
        for i in range(len(trade_day_range)-1):
            ret_code, content = quote_context.get_history_kline(self.stockcode, trade_day_range[i],
                                                                trade_day_range[i+1], ktype='K_1M')
            # print(content.columns)   # 上面接口返回的是pandas的dataframe对象
            # """
            # Index(['code', 'time_key', 'open', 'close', 'high', 'low', 'volume','turnover'], dtype='object')
            # """
            if ret_code != RET_OK:
                print("RTDataTest: error, msg: %s" % content)
                return RET_ERROR, content
            for _, row in content.iterrows():
                if row['time_key'] not in self.__time_array:
                    # print("%s %s" % (row['time_key'], round(row['open'], 3)))
                    db.table("KLine1M") \
                      .insert(StockCode=row['code'], Open=round(row['open'], 3), High=round(row['high'], 3),
                              Low=round(row['low'], 3), Close=round(row['close'], 3), Volume=row['volume'],
                              Turnover=row['turnover'], KLTime=row['time_key']) \
                      .execute(commit_at_once=False)
                    count += 1
        db.commit()
        print("%d rows inserted!" % count)
        db.close()

    def get_kline_view(self, from_time, to_time, title):
        """绘制某时段的K线"""
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_axes([0.1, 0.4, 0.85, 0.5])   # [left, bottom, width, height]
        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.3])
        qutotes = []
        index, indexs, volumns, lows, highs = 0, [], [], [], []
        for key in self.days_dict:
            if key > to_time:
                break
            if from_time < key < to_time:
                index += 1
                indexs.append(index)
                # d = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
                # d = index if 1=1 else mpf.date2num(d)
                d = index  # mpf.date2num(datetime.strptime(key, '%Y-%m-%d %H:%M:%S'))
                o = self.days_dict[key].Open
                l = self.days_dict[key].Low
                h = self.days_dict[key].High
                c = self.days_dict[key].Close
                volumns.append(self.days_dict[key].Volume)
                lows.append(l)
                highs.append(h)
                val = (d, o, c, h, l)
                qutotes.append(val)

        mpf.candlestick_ochl(ax1, qutotes, colorup='red', colordown='green')
        ax1.grid(True)
        ax1.axis([0, max(indexs)+1, min(lows)-0.1, max(highs)+0.1])
        ax2.grid(True)
        # ax1.autoscale_view()
        ax2.bar(indexs, volumns)
        # .axis([0, 6, 0, 20])是指定xy坐标的起始范围，它的参数是列表[xmin, xmax, ymin, ymax]
        ax2.axis([0, max(indexs)+1, 0, max(volumns)+10])
        # plt.title("%s ---- %s" % (from_time, to_time))
        plt.title(title)
        # ax.xaxis_date()
        # plt.show()
        f = datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        t = datetime.strptime(to_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        plt.savefig(r'D:\pics\%s -- %s.jpg' % (f, t))


if __name__ == '__main__':
    d = StockData('US.BABA')
    print(d)            # print data summary
    # print(d[0])         # print first line
    # print(d[0].Close)         # print second line
    # c = d.get_change_array()  # get change_array default close
    # print(c[0:3])               # print result
    # for x in d:
    #     print(x)
    # a = d[1]
    # print(a)
    # aa = d.get_keypoint_array("","")
    # print(aa)
    # foo = d.get_keypoint_list(todate='2017-11-27', rate=0.03)
    # for x in foo:
    #     print(x)
    # d.get_keypoint_report(foo)
    # d.get_data_view(foo)

    # kp_array = d.get_kp_array(rate=0.05)
    # for x in kp_array[1:]:
    #     f = d.add_mins_to_time(x.KLTime,-60)
    #     t = d.add_mins_to_time(x.KLTime,60)
    #     d.get_kline_view(f, t, x)
    #     print(x)

    # d.get_keypoint_report(kp_array)
    # 更新数据库
    # d.update_db()
    # d.get_kline_view('2017-02-13 10:34:00', '2017-02-13 12:34:00')

    # x = '2017-02-15 10:12:00'
    # f = d.add_mins_to_time(x, -60)
    # t = d.add_mins_to_time(x, 60)
    # d.get_kline_view(f, t, x)




