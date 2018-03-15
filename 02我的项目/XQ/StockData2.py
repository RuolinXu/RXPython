from SQLite3DB import SQLite3DB, DataCondition
from futuquant.futuquant.open_context import *
import pandas as pd
from pandas.core import datetools
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from XQ.RegUtil import *

DB_PATH = r'../FutuProject/StockDB.db'


def create_db_table():
    db = SQLite3DB(DB_PATH)
    db.open()
    # 创建1分钟线数据表
    # db.execute("""
    #     CREATE TABLE [KLine1M] (
    #           Id        INTEGER PRIMARY KEY AUTOINCREMENT,
    #           StockCode TEXT    NOT NULL,
    #           Open      REAL    NOT NULL,
    #           High      REAL    NOT NULL,
    #           Low       REAL    NOT NULL,
    #           Close     REAL    NOT NULL,
    #           KLTime    TEXT    NOT NULL,
    #           Turnover  REAL    NOT NULL,
    #           Volume    INTEGER NOT NULL
    #     )
    # """)  # 直接执行SQL语句，注意这里commit_at_once默认为True

    # 创建日线数据表
    # db.execute("""
    #     CREATE TABLE [KLineDAY] (
    #           Id        INTEGER PRIMARY KEY AUTOINCREMENT,
    #           StockCode TEXT    NOT NULL,
    #           Open      REAL    NOT NULL,
    #           High      REAL    NOT NULL,
    #           Low       REAL    NOT NULL,
    #           Close     REAL    NOT NULL,
    #           KLTime    TEXT    NOT NULL,
    #           Turnover  REAL    NOT NULL,
    #           Volume    INTEGER NOT NULL
    #     )
    # """)  # 直接执行SQL语句，注意这里commit_at_once默认为True

    # 创建股票池
    # db.execute("""
    #     CREATE TABLE [Symbols] (
    #           Id        INTEGER PRIMARY KEY AUTOINCREMENT,
    #           StockCode TEXT    NOT NULL UNIQUE,
    #           StockName TEXT    NOT NULL,
    #           lotSize   INTEGER NOT NULL,
    #           Market    TEXT    NOT NULL
    #     )
    # """)  # 直接执行SQL语句，注意这里commit_at_once默认为True

    # rs = db.execute("""
    # select max(KLTime) from %s where StockCode = '%s'
    # """ % ('KLine1M', 'US.BAB1'))
    db.close()  # 关闭数据库
    # return rs


class StockData2:
    """提供从数据库获取数据的方法"""
    _DB_PATH = r'../FutuProject/StockDB.db'

    @classmethod
    def kline_pd_from_db(cls, symbol, start='2000-01-01', end='2100-01-01', ktype='K_1M'):
        db = SQLite3DB(cls._DB_PATH)
        db.open()
        target_table = 'KLine1M' if ktype == 'K_1M' else 'KLineDAY'
        rs = db.table(target_table) \
            .select("Open", "Low", "High", "Close", "Volume", "Turnover", "KLTime") \
            .where(DataCondition(("=", "AND"), StockCode=symbol),
                   DataCondition((">", "AND"), KLTime=start),
                   DataCondition(("<", "AND"), KLTime=end)) \
            .fetchall()
        # 0=Open 1=Low 2=High 3=Close 4=Volume 5=Turnover 6=KLTime
        db.close()
        sorted_rs = sorted(rs, key=lambda a: a[6])  # sorted by time
        if len(sorted_rs) < 1:
            return None
        data_frame = pd.DataFrame(sorted_rs, columns=["Open", "Low", "High", "Close", "Volume", "Turnover", "KLTime"])
        data_frame['KLTime'] = datetools.to_datetime(data_frame['KLTime'], format='%Y-%m-%d %H:%M:%S')
        return data_frame.set_index(['KLTime'])

    @classmethod
    def _update_db_1M(cls, symbol):
        """调用富途接口，把数据添加到数据库 """
        quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)
        market = symbol.split('.')[0]
        tmp_pd = cls.kline_pd_from_db(symbol, ktype='K_1M')
        if tmp_pd is not None:
            time_array = [str(x) for x in tmp_pd.index.to_pydatetime()]
        else:
            time_array = []
        if len(time_array) < 1:
            from_date = '2017-01-01'
        else:
            from_date = time_array[-1][0:10]
        to_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        # from_date = '2017-09-21'
        # to_date = '2017-09-26'
        ret_code, content = quote_context.get_trading_days(market, from_date, to_date)
        if ret_code != RET_OK:
            print("RTDataTest: error, msg: %s" % content)
            return RET_ERROR, content
        trade_days = sorted(content)  # 排序交易日
        trade_day_range = trade_days[::3]  # 每次取3天
        if trade_day_range[-1] != trade_days[-1]:
            trade_day_range.append(trade_days[-1])
        db = SQLite3DB(cls._DB_PATH)
        db.open()
        count = 0
        for i in range(len(trade_day_range) - 1):
            ret_code, content = quote_context.get_history_kline(symbol, trade_day_range[i],
                                                                trade_day_range[i + 1], ktype='K_1M')
            print("%s from %s to %s" % (symbol, trade_day_range[i], trade_day_range[i + 1]))
            # print(content.columns)   # 上面接口返回的是pandas的dataframe对象
            # """
            # Index(['code', 'time_key', 'open', 'close', 'high', 'low', 'volume','turnover'], dtype='object')
            # """
            if ret_code != RET_OK:
                print("RTDataTest: error, msg: %s" % content)
                return RET_ERROR, content
            tmplist = []
            for _, row in content.iterrows():
                if row['time_key'] not in time_array and row['time_key'] not in tmplist:
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
        """ % symbol)
        print("%s %d rows inserted!" % (symbol, count))
        db.close()

    @classmethod
    def _update_db_DAY(cls, symbol):
        """调用富途接口，把数据添加到数据库 """
        quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)
        tmp_pd = cls.kline_pd_from_db(symbol, ktype='K_DAY')
        if tmp_pd is not None:
            time_array = [str(x) for x in tmp_pd.index.to_pydatetime()]
        else:
            time_array = []
        if len(time_array) < 1:
            from_date = '1990-01-01'
        else:
            from_date = time_array[-1][0:10]
        to_date = '2100-01-01'  # (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        db = SQLite3DB(cls._DB_PATH)
        db.open()
        count = 0

        ret_code, content = quote_context.get_history_kline(symbol, from_date, to_date, ktype='K_DAY')
        print("Update KLineDAY %s from %s to %s" % (symbol, from_date, to_date))
        # print(content.columns)   # 上面接口返回的是pandas的dataframe对象
        # """
        # Index(['code', 'time_key', 'open', 'close', 'high', 'low', 'volume','turnover'], dtype='object')
        # """
        if ret_code != RET_OK:
            print("RTDataTest: error, msg: %s" % content)
            return RET_ERROR, content
        tmplist = []
        for _, row in content.iterrows():
            if row['time_key'] not in time_array and row['time_key'] not in tmplist:
                # print("%s %s" % (row['time_key'], round(row['open'], 3)))
                db.table("KLineDAY") \
                    .insert(StockCode=row['code'], Open=round(row['open'], 3), High=round(row['high'], 3),
                            Low=round(row['low'], 3), Close=round(row['close'], 3), Volume=row['volume'],
                            Turnover=row['turnover'], KLTime=row['time_key']) \
                    .execute(commit_at_once=False)
                tmplist.append(row['time_key'])
                count += 1
        db.commit()
        db.execute("""
            delete from KLineDAY where id in (
              select max(id) from KLineDAY 
              group by StockCode,KLTime 
              having (StockCode='%s' and count(*) >1)
            )
        """ % symbol)
        print("%s %d rows inserted!" % (symbol, count))
        db.close()

    @classmethod
    def update_db_kline1M(cls):
        symbol_list = ['US.BABA', 'US.NVDA', 'US.AAOI']
        for x in symbol_list:
            cls._update_db_1M(x)
        pass

    @classmethod
    def update_db_klineDAY(cls):
        symbol_list = ['US.BABA', 'US.NVDA', 'US.INTC', 'US.AAPL', 'US.TSLA', 'US.GS', 'US.JPM', 'US.BZUN']
        for x in symbol_list:
            cls._update_db_DAY(x)
        pass

    @classmethod
    def view_kline(cls, kline_pd, start='2000-01-01', end='2100-01-01', title=''):
        """绘制某时段的K线"""
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_axes([0.1, 0.4, 0.85, 0.5])  # [left, bottom, width, height]
        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.3])

        # df = self.stockdata_df.query("index > @from_time and index < @to_time ")
        df = kline_pd[(start < kline_pd.index) & (kline_pd.index < end)]
        i_length = len(df)
        qutotes = [(i, x[1]['Open'], x[1]['Close'], x[1]['High'], x[1]['Low'])
                   for i, x in zip(range(i_length), df.iterrows())]
        mpf.candlestick_ochl(ax1, qutotes, colorup='red', colordown='green')
        # mpf.candlestick2_ochl(ax=ax1,
        #                   opens=df["Open"].values, closes=df["Close"].values,
        #                   highs=df["High"].values, lows=df["Low"].values,
        #                   width=0.75, colorup='r', colordown='g', alpha=0.75)

        ax1.grid(True)
        ax1.axis([-1, i_length + 1, df["Low"].min() - 0.1, df["High"].max() + 0.1])
        ax2.grid(True)
        # ax1.autoscale_view()
        ax2.bar(range(0, i_length), df["Volume"].values)
        # .axis([0, 6, 0, 20])是指定xy坐标的起始范围，它的参数是列表[xmin, xmax, ymin, ymax]
        ax2.axis([-1, i_length + 1, 0, df["Volume"].max() + 10])
        # plt.title("%s ---- %s" % (from_time, to_time))
        plt.title(title)
        _, y = regress_y(df['Close'], mode=False, show=False)
        # print(y)
        plt.sca(ax1)
        plt.plot(range(0, len(y)), y)
        plt.show()
        # f = datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        # t = datetime.strptime(to_time, '%Y-%m-%d %H:%M:%S').strftime('%m-%d %H_%M')
        # plt.savefig(r'D:\pics\%s -- %s.jpg' % (f, t))
        pass

    @classmethod
    def statistics_kline(cls, kline_pd):
        pass


if __name__ == '__main__':
    # create_db_table()
    StockData2.update_db_kline1M()
    # StockData2.update_db_klineDAY()
    # pd = StockData2.kline_pd_from_db('US.BABA', ktype='K_DAY')
    # StockData2.view_kline(pd, start='2017-05-01')
    pass
