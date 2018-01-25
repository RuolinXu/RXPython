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
        self.stockdata_df = self.__dataFrame_from_db()
        self.time_array = self.stockdata_df.index.values
        self.stockdata_od = self.__orderdict_from_db()
        # self.__sorted_rs = self.__load_sorted_rs()
        # self.days_dict = self.__init_days_dict()
        # self.__sorted_rs = None  # free some memory

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
        if len(sorted_rs) < 1:
            return None
        data_frame = pd.DataFrame(sorted_rs, columns=["Open", "Low", "High", "Close", "Volume", "Turnover", "KLTime"])
        return data_frame.set_index(['KLTime'])

    def __orderdict_from_db(self):
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
        if len(sorted_rs) < 1:
            return None
        stock_nametuple = namedtuple(self.stockcode.split('.')[1],
                                     ('Open', 'Low', 'High', 'Close', 'Volume', 'Turnover', 'KLTime'))
        # 0=Open 1=Low 2=High 3=Close 4=Volume 5=Turnover 6=KLTime
        stockdata_od = OrderedDict((x[6], stock_nametuple(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
                                for x in sorted_rs)

        return stockdata_od

    def __iter__(self):
        for key in self.stockdata_od:
            yield self.stockdata_od[key]

    def __getitem__(self, ind):
        time_key = self.time_array[ind]
        return self.stockdata_od[time_key]

    def __str__(self):
        if len(self.time_array) < 1:
            return 'No Data!'
        start = self.time_array[1]
        end = self.time_array[-1]
        return str(self.stockdata_od[start]) + '\n...\n' + str(self.stockdata_od[end])

    def update_db(self):
        """调用富途接口，把数据添加到数据库 """
        quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)
        market = self.stockcode.split('.')[0]
        if len(self.time_array) < 1:
            from_date = '2017-01-01'
        else:
            from_date = self.time_array[-1][0:10]
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
                if row['time_key'] not in self.time_array and row['time_key'] not in tmplist:
                    # print("%s %s" % (row['time_key'], round(row['open'], 3)))
                    db.table("KLine1M") \
                      .insert(StockCode=row['code'], Open=round(row['open'], 3), High=round(row['high'], 3),
                              Low=round(row['low'], 3), Close=round(row['close'], 3), Volume=row['volume'],
                              Turnover=row['turnover'], KLTime=row['time_key']) \
                      .execute(commit_at_once=False)
                    tmplist.append(row['time_key'])
                    count += 1
        db.commit()
        db.execute("""
        
        """)
        print("%d rows inserted!" % count)
        db.close()

    def get_kline_view(self, from_time, to_time, title):
        """绘制某时段的K线"""
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_axes([0.1, 0.4, 0.85, 0.5])   # [left, bottom, width, height]
        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.3])

        df = self.stockdata_df.query("index > @from_time and index < @to_time ")
        i_length = len(df)
        qutotes = [(i, x[1]['Open'], x[1]['Close'], x[1]['High'], x[1]['Low'])
                   for i, x in zip(range(i_length), df.iterrows())]
        mpf.candlestick_ochl(ax1, qutotes, colorup='red', colordown='green')
        # mpf.candlestick2_ochl(ax=ax1,
        #                   opens=df["Open"].values, closes=df["Close"].values,
        #                   highs=df["High"].values, lows=df["Low"].values,
        #                   width=0.75, colorup='r', colordown='g', alpha=0.75)

        ax1.grid(True)
        ax1.axis([-1, i_length+1, df["Low"].min()-0.1, df["High"].max()+0.1])
        ax2.grid(True)
        # ax1.autoscale_view()
        ax2.bar(range(0,i_length), df["Volume"].values)
        # .axis([0, 6, 0, 20])是指定xy坐标的起始范围，它的参数是列表[xmin, xmax, ymin, ymax]
        ax2.axis([-1, i_length+1, 0, df["Volume"].max()+10])
        # plt.title("%s ---- %s" % (from_time, to_time))
        plt.title(title)
        plt.show()
        # f = datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        # t = datetime.strptime(to_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        # plt.savefig(r'D:\pics\%s -- %s.jpg' % (f, t))

    def foo(self, fromdate, todate):
        """
        总成交量TV 上涨成交量UV 下跌成交量DV 上涨成交均价UP 下跌成交均价DP 上涨数量UC 下跌数量 DC
        高低价价差平均HLD  上涨每分成交量UVHLP平均  下跌每分成交量
        上涨每分成交量UVHLP 最大前十明细   （时间 VHLP  HL Volumn UP  ）
        """
        # dd = self.stockdata_od[1]

        df = self.stockdata_df[fromdate: todate]

        df['VHLP'] = df.eval('Volume / 100*(High-Low)')    # 每分钱成交量
        # df['AVGPrice'] = df.eval('Turnover / Volume')    # 成交均价
        print(df)
        print(df[df.Close > df.Open].sort_values('VHLP', ascending=False).ix[:10, ])
        # print(df)  ['Open', 'Close', 'VHLP']




if __name__ == '__main__':
    # d = StockData('US.BABA')
    d = StockData('US.BABA')
    # print(d.time_array[1])                    # print data summary
    # d.update_db()
    d.foo('2017-01-31 15:39:00', '2017-02-01 09:39:00')

    # print(d.stockdata_df.loc['2017-01-31 09:39:00']['Turnover'])
    # df = d.stockdata_df
    # print(df[df.index < '2017-02-01 09:31:00'])    #
    # print(df[('2017-01-31 10:31:00' < df.index) & (df.index < '2017-02-01 09:31:00')])  # 选择区间
    # df2 = df[('2017-01-31 10:31:00' < df.index) & (df.index < '2017-02-01 09:31:00')]
    # print(df2.High.mean())

    # df3 = d.stockdata_df.query("index > '2017-01-31 10:31:00' and index < '2017-01-31 12:31:00' ")
    # _high = df3.High.max()
    # df4 = df3.query("High == @_high").index
    # _high_time = df4[0]
    # print(df3)
    # df3 = df3.round(3)
    # qutotes = [(i, x[1]['Open'], x[1]['Close'], x[1]['High'], x[1]['Low'])
    #            for i, x in zip(range(len(df3)), df3.iterrows())]
    # print(df3.loc['2017-01-31 11:23:00'])
    # print(_high_time)
    # print(df3)
    # d.get_kline_view('2017-01-31 10:31:00', '2017-01-31 12:31:00','xxx')

    # print(d[0])         # print first line
    # print(d[0].Close)         # print second line
    # c = d.get_change_array()  # get change_array default close
    # print(c[0:3])               # print result
    # for x in d.stockdata_od:
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




