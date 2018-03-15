from collections import namedtuple
from collections import OrderedDict
from SQLite3DB import SQLite3DB, DataCondition
import datetime
from futuquant.futuquant.open_context import *
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import pandas as pd


DB_PATH = r'./StockDB.db'


class StockData(object):
    """history data from DB"""
    def __init__(self, stockcode, fromdate='2000-01-01', todate='2100-01-01'):
        self.stockcode = stockcode
        self.fromdate = fromdate
        self.todate = todate
        self.stockdata_df = self.__dataframe_from_db()
        self.time_array = [str(x) for x in self.stockdata_df.index.to_pydatetime()]  # self.stockdata_df.index.values
        self.stockdata_od = self.__orderdict_from_db()
        # self.__sorted_rs = self.__load_sorted_rs()
        # self.days_dict = self.__init_days_dict()
        # self.__sorted_rs = None  # free some memory

    def __dataframe_from_db(self):
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
        data_frame['KLTime'] = pd.to_datetime(data_frame['KLTime'], format='%Y-%m-%d %H:%M:%S')
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
        delete from KLine1M where id in (
          select max(id) from KLine1M 
          group by StockCode,KLTime 
          having (StockCode='%s' and count(*) >1)
        )
        """ % self.stockcode)
        print("%s %d rows inserted!" % (self.stockcode, count))
        db.close()

    def get_kline_view(self, from_time, to_time, title):
        """绘制某时段的K线"""
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_axes([0.1, 0.4, 0.85, 0.5])   # [left, bottom, width, height]
        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.3])

        # df = self.stockdata_df.query("index > @from_time and index < @to_time ")
        df = self.stockdata_df[from_time: to_time]
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
        高低价价差平均HLD  上涨每分成交量UVHLP平均  下跌每分成交量DVHLP平均
        上涨每分成交量UVHLP 最大前十明细   （时间 VHLP  HL Volumn UP  ）
        """
        pd.options.mode.chained_assignment = None  # default='warn'
        df = self.stockdata_df[fromdate: todate]
        df['VHLP'] = round(df.eval('Volume / (100*(High-Low))'), 1)     # 每分钱成交量
        df['AvgP'] = round(df.eval('Turnover / Volume'), 3)             # 成交均价
        df['HLD'] = df.eval('High-Low')                                 # 振幅
        df['IsUp'] = df.eval('Close > Open')
        df['UpShadow'] = df.apply(lambda x: (x.High-x.Close) if x.IsUp else (x.High - x.Open), axis=1)
        df['DownShadow'] = df.apply(lambda x: (x.Low - x.Open) if x.IsUp else (x.Low - x.Close), axis=1)
        df['Solid'] = df.eval('Close - Open')

        TV = df['Volume'].sum()                                         # 总成交量
        Udf, Ddf = df[df.Close > df.Open], df[df.Close < df.Open]       # 分割阳线 阴线
        UC, DC = Udf.shape[0], Ddf.shape[0]                             # 计算数量
        UV, DV = Udf['Volume'].sum(), Ddf['Volume'].sum()               # 计算阳线 阴线成交量总量
        UP = round(Udf['Turnover'].sum() / Udf['Volume'].sum(), 3)      # 阳线成交均价
        DP = round(Ddf['Turnover'].sum() / Ddf['Volume'].sum(), 3)      # 阴线成交均价
        HLD = round(df['HLD'].mean(), 3)                                # 高低价差平均
        UVHLP, DVHLP = round(Udf['VHLP'].mean(), 3), round(Ddf['VHLP'].mean(), 3)   # 阳线 阴险高低价差平均
        UpShadowS, DownShadowS = df['UpShadow'].sum(), df['DownShadow'].sum()       # 上影线 下影线加总
        UpSolid, DownSolid = Udf['Solid'].sum(), Ddf['Solid'].sum()     # 实体区间之和
        print('%s  %s - %s 的分析：' % (self.stockcode, df.index.values[0], df.index.values[-1]))
        print('*'*100)
        # print('总成交量TV\t上涨成交量UV\t下跌成交量DV\t上涨成交均价UP\t下跌成交均价DP\t上涨数量UC\t下跌数量DC')
        print('%12s\t%12s\t%12s\t%8s\t%10s\t%10s\t%10s'
              % ('Total_Volume', 'Up_Volume', 'Down_Volume',
                 'Up_Price', 'Down_Price', 'Up_Count', 'Down_Count'))
        print('%12d\t%12d\t%12d\t%8.3f\t%10.3f\t%10d\t%10d\n' % (TV, UV, DV, UP, DP, UC, DC))
        print('%12s\t%14s\t%10s\t%10s' % ('UpShadow_Sum', 'DownShadow_Sum', 'Up_solid', 'Down_solid'))
        print('%12.3f\t%14.3f\t%10.3f\t%10.3f\n' % (UpShadowS, DownShadowS, UpSolid, DownSolid))
        print('高低价价差平均HLD  上涨每分成交量UVHLP平均  下跌每分成交量DVHLP平均')
        print('{}\t\t\t\t{}\t\t\t\t{}'.format(HLD, UVHLP, DVHLP))
        # print('%12s\t%17s\t%17s' % ('HL Range Avg', 'Up Penny Volume', 'Down Penny Volume'))
        # print('%12.3f\t%12.3f\t%12.3f' % (HLD, UVHLP, DVHLP))
        print('*' * 100)
        if UC > 10:
            print('上涨每分成交量 UVHLP 前10：（查看是否入货）')
            print(Udf.sort_values('VHLP', ascending=False).ix[:10, ['VHLP', 'Volume', 'AvgP', 'HLD']])
            print('*' * 100)
        if DC > 10:
            print('下跌每分成交量 UVHLP 前10：（查看是否出货）')
            print(Ddf.sort_values('VHLP', ascending=False).ix[:10, ['VHLP', 'Volume', 'AvgP', 'HLD']])
            print('*' * 100)
        if df.shape[0] > 10:
            print('VHLP最小的前10：（查看是否拉升或打压）')
            print(df.sort_values('VHLP', ascending=True).ix[:10, ['VHLP', 'Volume', 'HLD', 'IsUp', 'Open', 'Close']])
            print('*' * 100)
            print('上影线最大的前10：（查看上涨意愿）')
            print(df.sort_values('UpShadow', ascending=False).ix[:10, ['VHLP', 'UpShadow', 'High', 'AvgP', 'IsUp']])
            print('*' * 100)
            print('下影线最大的前10：（查看下跌意愿）')
            df1 = df.sort_values('DownShadow', ascending=True).ix[:10, ['VHLP', 'DownShadow', 'Low', 'AvgP', 'IsUp']]
            print(df1.sort_index(ascending=False))
        # print(df[df.Close > df.Open].sort_values('VHLP', ascending=False).ix[:10, ])
        # print(df)  ['Open', 'Close', 'VHLP']


if __name__ == '__main__':
    # d = StockData('US.NVDA')BABA
    d = StockData('US.AAOI')
    # t1 = d.stockdata_df.index.to_pydatetime()
    # print(str(t1[0]))                    # print data summary

    # d.update_db()
    d.foo('2018-02-23 09:30:00', '2018-03-15 16:00:00')
    # d.get_kline_view('2018-01-30 09:30:00', '2018-01-30 16:00:00', '')

    # d1 = d.stockdata_df.resample('5T').mean()  # T分钟 D日
    # print(d1['Open'])

    # d2 = d.stockdata_df  # T分钟 D日
    # d2['pct_change'] = d2['Close'].pct_change()
    # print(d2.filter(['Close','Open','pct_change']))
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




