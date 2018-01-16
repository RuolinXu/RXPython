from collections import namedtuple
from collections import OrderedDict
from SQLite3DB import SQLite3DB, DataCondition
import datetime
from futuquant.futuquant.open_context import *
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import pandas as pd
import numpy as np


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
        self.stockdata_df = self.__dataFrame_from_db()

    def __dataFrame_from_db(self):
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
        data_frame = pd.DataFrame(sorted_rs, columns=["Open", "Low", "High", "Close", "Volume", "Turnover", "KLTime"])
        return data_frame.set_index(['KLTime'])

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
        if len(self.__sorted_rs) < 1:
            return None
        stock_nametuple = namedtuple(self.stockcode.split('.')[1],
                                     ('Open', 'Low', 'High', 'Close', 'Volume', 'Turnover', 'KLTime'))
        # 0=Open 1=Low 2=High 3=Close 4=Volume 5=Turnover 6=KLTime
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
        if len(self.__time_array) < 1:
            return 'No Data!'
        start = self.__time_array[1]
        end = self.__time_array[-1]
        return str(self.days_dict[start]) + '\n...\n' + str(self.days_dict[end])

    def update_db(self):
        """调用富途接口，把数据添加到数据库 """
        quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)
        market = self.stockcode.split('.')[0]
        if len(self.__time_array) < 1:
            from_date = '2017-01-01'
        else:
            from_date = self.__time_array[-1][0:10]
        to_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        # from_date = '2017-09-21'
        # to_date = '2017-09-26'
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
            print("from %s to %s" % (trade_day_range[i], trade_day_range[i+1]))
            # print(content.columns)   # 上面接口返回的是pandas的dataframe对象
            # """
            # Index(['code', 'time_key', 'open', 'close', 'high', 'low', 'volume','turnover'], dtype='object')
            # """
            if ret_code != RET_OK:
                print("RTDataTest: error, msg: %s" % content)
                return RET_ERROR, content
            tmplist = []
            for _, row in content.iterrows():
                if row['time_key'] not in self.__time_array and row['time_key'] not in tmplist:
                    # print("%s %s" % (row['time_key'], round(row['open'], 3)))
                    db.table("KLine1M") \
                      .insert(StockCode=row['code'], Open=round(row['open'], 3), High=round(row['high'], 3),
                              Low=round(row['low'], 3), Close=round(row['close'], 3), Volume=row['volume'],
                              Turnover=row['turnover'], KLTime=row['time_key']) \
                      .execute(commit_at_once=False)
                    tmplist.append(row['time_key'])
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
    # d = StockData('US.BABA')
    d = StockData('US.NVDA')
    print(d)                    # print data summary
    # d.update_db()

    # print(d.stockdata_df.loc['2017-01-31 09:39:00']['Turnover'])
    df = d.stockdata_df
    # print(df[df.index < '2017-02-01 09:31:00'])    #
    print(df[('2017-01-31 10:31:00' < df.index) & (df.index < '2017-02-01 09:31:00')])  # 选择区间
    df2 = df[('2017-01-31 10:31:00' < df.index) & (df.index < '2017-02-01 09:31:00')]
    print(df2.High.mean())
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




