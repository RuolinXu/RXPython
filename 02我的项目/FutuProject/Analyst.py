from functools import reduce
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from RedisHelper import *


class AnalystBase:
    def __init__(self, stockdata):
        self.stockdata = stockdata
        self.days_dict = stockdata.days_dict
        self._time_array = [x for x in self.days_dict.keys()]
        self.__kpm_cache = None
        self.__cache = None

    @classmethod
    def _get_change_array(cls, tar, base):
        pp_array = [(p1, p2) for p1, p2 in zip(base[:-1], tar[1:])]
        change_array = list(map(lambda pp: reduce(lambda a, b: (b - a) / a, pp), pp_array))
        change_array.insert(0, 0)
        return change_array

    @classmethod
    def _get_interval_array(cls, tar, base):
        pp_array = [(p1, p2) for p1, p2 in zip(base[:-1], tar[1:])]
        interval_array = list(map(lambda pp: reduce(lambda a, b: b - a, pp), pp_array))
        interval_array.insert(0, 0)
        return interval_array

    @staticmethod
    def _gettime_add_ind(current, ind, period=390, shour=9, smin=30):
        current_time = datetime.strptime(current, '%Y-%m-%d %H:%M:%S')
        start_time = datetime(current_time.year, current_time.month, current_time.day, shour, smin)
        delta = int((current_time - start_time).seconds / 60)
        days = int((ind + delta) / period)
        mins = int((ind + delta) % period)
        if ind < 0 and abs(ind) > delta:
            days -= 1
        time_tar = start_time + timedelta(days=days, minutes=mins)
        return time_tar.strftime('%Y-%m-%d %H:%M:%S')

    def get_kpm_array(self, todate='2100-01-01', rate=0.03):
        if self.__kpm_cache is None:
            self.__kpm_cache = []
        if len(self.__kpm_cache) < 2:
            self.__kpm_cache = self._get_kp_array(todate=todate, rate=rate)
        else:
            fromdate = self.__kpm_cache[-2].KLTime
            tmp = self._get_kp_array(fromdate=fromdate, todate=todate, rate=rate)
            if len(tmp) > 0:
                self.__kpm_cache.pop()
                for x in tmp[1:]:
                    self.__kpm_cache.append(x)
        return self.__kpm_cache

    def get_kp_array_cache(self, kltime):
        if self.__cache is None:
            self.__cache = self._get_kp_array()
        rst = list(filter(lambda x: x.KLTime < kltime, self.__cache))
        return sorted(rst, key=lambda x: x.KLTime)

    def _get_kp_array(self, fromdate='2000-01-01', todate='2100-01-01', rate=0.03):
        """返回走势骨架
        :param fromdate:
        :param todate:
        :param rate:涨幅基数
        :return: namedtuple_list('Ind', 'Price', 'KLTime', 'D')
        """
        p_nametuple = namedtuple('stock', ('Ind', 'Price', 'KLTime', 'D'))
        kp_array = []
        for key in self.days_dict:
            if key > todate:
                break
            if fromdate < key < todate:
                _ind = self._time_array.index(key)
                _open = self.days_dict[key].Open
                _low = self.days_dict[key].Low
                _high = self.days_dict[key].High
                _ktime = self.days_dict[key].KLTime
                if len(kp_array) == 0:
                    kp_array.append(p_nametuple(_ind, _open, _ktime, 0))
                    continue
                if kp_array[-1].D == 0:
                    if _high > (kp_array[-1].Price * (1 + rate)):
                        kp_array.append(p_nametuple(_ind, _high, _ktime, 1))
                        continue
                    if _low < (kp_array[-1].Price * (1 - rate)):
                        kp_array.append(p_nametuple(_ind, _low, _ktime, -1))
                        continue
                if kp_array[-1].D == 1:
                    if _high > kp_array[-1].Price:
                        kp_array[-1] = p_nametuple(_ind, _high, _ktime, 1)
                        continue
                    elif _low < (kp_array[-1].Price * (1 - rate)):
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

        time_str = self._gettime_add_ind(kp_array[-1].KLTime, time_delta)
        print("目标价%f,时间点%s" % (price_tar, time_str))
        plt.figure(figsize=(16, 9))
        plt.plot(ind_array, price_array)
        plt.show()


class AnalystA(AnalystBase):
    """未完成
    大于前高点 某个百分点 卖出
    小于前高点 某个百分点 买入
    """
    def __init__(self, stockdata, b_rate=-0.09, s_rate=0.15):
        AnalystBase.__init__(self, stockdata)

    def can_buy(self, kltime):
        return self.__buy_sell_report(what='buy', kltime=kltime)

    def can_sell(self, kltime):
        return self.__buy_sell_report(what='sell', kltime=kltime)

    def __buy_sell_report(self, what, kltime='2100-01-01'):

        kp_array = self.get_kp_array_cache(kltime)

        if len(kp_array) < 3:
            print("Not enough data, keep watching...")
            return

        # 获取前高点
        if kp_array[-1].D == 1:
            kp_buy_ref = kp_array[-1]   # 买点依据上一高点
            kp_sell_ref = kp_array[-2]  # 卖点依据上一低点
        else:
            kp_buy_ref = kp_array[-2]
            kp_sell_ref = kp_array[-1]

        price_array = np.array([x.Price for x in kp_array])
        ind_array = np.array([x.Ind for x in kp_array])
        change_array = self._get_change_array(price_array, price_array)
        interval_array = self._get_interval_array(ind_array, ind_array)
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
        bt_min, bt_avg, bt_max = down_interval_array.min(), down_interval_array.mean(), down_interval_array.max()
        bc_min, bc_avg, bc_max = down_change_array.min(), down_change_array.mean(), down_change_array.max()
        st_min, st_avg, st_max = up_interval_array.min(), up_interval_array.mean(), up_interval_array.max()
        sc_min, sc_avg, sc_max = up_change_array.min(), up_change_array.mean(), up_change_array.max()

        def report():
            if len(kp_array) < 3:
                print("Not enough data, keep watching...")
                return
            print("Ref:%s %.03f. You can buy at" % (kp_buy_ref.KLTime, kp_buy_ref.Price))
            print("if before %s and price <= %.03f, you can buy"
                  % (self._gettime_add_ind(kp_buy_ref.KLTime, bt_min), (kp_buy_ref.Price * (1 + bc_min))))
            print("if between %s and %s, and price < %.03f, you can buy"
                  % (self._gettime_add_ind(kp_buy_ref.KLTime, bt_min),
                     self._gettime_add_ind(kp_buy_ref.KLTime, bt_avg), (kp_buy_ref.Price * (1 + bc_avg))))
            print("if between %s and %s, and price < %.03f, you can buy"
                  % (self._gettime_add_ind(kp_buy_ref.KLTime, bt_avg),
                     self._gettime_add_ind(kp_buy_ref.KLTime, bt_max), (kp_buy_ref.Price * (1 + bc_max))))
            print("if after %s and price <= %.03f, you can buy"
                  % (self._gettime_add_ind(kp_buy_ref.KLTime, bt_max), (kp_buy_ref.Price * (1 + bc_max / 2))))
            print("Ref:%s %.03f. You can sell at" % (kp_sell_ref.KLTime, kp_sell_ref.Price))
            print("if before %s and price >= %.03f, you can sell"
                  % (self._gettime_add_ind(kp_sell_ref.KLTime, st_min), (kp_sell_ref.Price * (1 + sc_max))))
            print("if between %s and %s, and price > %.03f, you can sell"
                  % (self._gettime_add_ind(kp_sell_ref.KLTime, st_min),
                     self._gettime_add_ind(kp_sell_ref.KLTime, st_avg), (kp_sell_ref.Price * (1 + sc_avg))))
            print("if between %s and %s, and price > %.03f, you can sell"
                  % (self._gettime_add_ind(kp_sell_ref.KLTime, st_avg),
                     self._gettime_add_ind(kp_sell_ref.KLTime, st_max), (kp_sell_ref.Price * (1 + sc_min))))
            print("if after %s and price >= %.03f, you can sell"
                  % (self._gettime_add_ind(kp_sell_ref.KLTime, st_max), (kp_sell_ref.Price * (1 + sc_min / 2))))
            print("Other conditions please wait !!!")

        if self._time_array[-1] < kltime:
            cur_price = self.stockdata[-1].Close
        else:
            cur_price = self.days_dict[kltime].Close

        def can_buy():
            if len(kp_array) < 3:
                return False
            if kltime <= self._gettime_add_ind(kp_buy_ref.KLTime, bt_min) and cur_price <= (kp_buy_ref.Price * (1 + bc_min)):
                return True
            if self._gettime_add_ind(kp_buy_ref.KLTime, bt_min) < kltime < self._gettime_add_ind(kp_buy_ref.KLTime, bt_avg) \
                    and cur_price < (kp_buy_ref.Price * (1 + bc_avg)):
                return True
            if self._gettime_add_ind(kp_buy_ref.KLTime, bt_avg) <= kltime < self._gettime_add_ind(kp_buy_ref.KLTime, bt_max) \
                    and cur_price < (kp_buy_ref.Price * (1 + bc_max)):
                return True
            if self._gettime_add_ind(kp_buy_ref.KLTime, bt_max) <= kltime and cur_price <= (kp_buy_ref.Price * (1 + bc_max / 2)):
                return True
            return False

        def can_sell():
            if len(kp_array) < 3:
                return False
            if kltime <= self._gettime_add_ind(kp_sell_ref.KLTime, st_min) and cur_price >= (kp_sell_ref.Price * (1 + sc_max)):
                return True
            if self._gettime_add_ind(kp_sell_ref.KLTime, st_min) < kltime < self._gettime_add_ind(kp_sell_ref.KLTime, st_avg) \
                    and cur_price > (kp_sell_ref.Price * (1 + sc_avg)):
                return True
            # if self.__gettime_add_ind(kp_sell_ref.KLTime, st_avg) <= kltime < self.__gettime_add_ind(kp_sell_ref.KLTime, st_max) \
            #         and cur_price > (kp_sell_ref.Price * (1 + sc_min)):
            #     return True
            # if self.__gettime_add_ind(kp_sell_ref.KLTime, st_max) <= kltime and cur_price >= (kp_sell_ref.Price * (1 + sc_min / 2)):
            #     return True
            return False

        if what == 'buy':
            b = can_buy()
            s = can_sell()
            if b is True and s is False:
                return True
            else:
                return False
        elif what == 'sell':
            b = can_buy()
            s = can_sell()
            if s is True and b is False:
                return True
            else:
                return False
        else:
            return report()

    def get_report(self, kltime='2100-01-01'):
        self.__buy_sell_report(what='report', kltime=kltime)


class AnalystB:
    """
    大于前高点 某个百分点 卖出
    小于前高点 某个百分点 买入
    """
    def __init__(self, g_cache, stockdata, days, b_rate=-0.09, s_rate=0.15):
        self.data_df = stockdata.stockdata_df
        self.b_rate = b_rate
        self.s_rate = s_rate
        self.days = days*-1
        self.__status_cache_key = ""
        self.__status_cache = g_cache
        # self.__redis = RedisHelper()

    def can_buy(self, kltime):
        return self.__buy_sell_report(what='buy', kltime=kltime)

    def can_sell(self, kltime):
        return self.__buy_sell_report(what='sell', kltime=kltime)

    def __getstatus_cache(self, key):
        if key not in self.__status_cache:
            self.__status_cache[key] = self.__getstatus(key[1])
        # a = self.__redis.get(key)
        return self.__status_cache[key]

    def __getstatus(self, time_key):
        t_now = datetime.strptime(time_key, '%Y-%m-%d')
        from_time = t_now + timedelta(days=self.days)
        from_t = from_time.strftime('%Y-%m-%d %H:%M:%S')
        to_t = t_now.strftime('%Y-%m-%d %H:%M:%S')
        range_df = self.data_df.query("index > @from_t and index < @to_t ")
        if len(range_df.index) < 1:
            # print("Not enough data, keep watching...%s" % kltime)
            return 0, ""
        # 获取前高点
        _high = range_df.High.max()
        _high_time = range_df.query("High == @_high").index[0]
        return _high, _high_time

    def __buy_sell_report(self, what, kltime):
        # 获取前高点
        _high, _high_time = self.__getstatus_cache((str(self.days), kltime[0:10]))

        def report():
            print("前高点为:%s %.03f." % (_high_time, _high))
            print("when price <= %.03f, you can buy \nwhen price >= %.03f, you can sell"
                  % ((_high * (1 + self.b_rate)), (_high * (1 + self.s_rate))))

        # 取当前价 防止下标溢出
        cur_price = round(self.data_df.loc[kltime].Close, 3)
        # print('high = %s, b_rate = %s cur_price:%s' % (_high, self.b_rate, cur_price))
        def can_buy():
            if cur_price <= (_high * (1 + self.b_rate)):
                return True
            return False

        def can_sell():
            if cur_price >= (_high * (1 + self.s_rate)):
                return True
            return False

        if what == 'buy':
            b = can_buy()
            s = can_sell()
            if b is True and s is False:
                return True
            else:
                return False
        elif what == 'sell':
            b = can_buy()
            s = can_sell()
            if s is True and b is False:
                return True
            else:
                return False
        else:
            return report()

    def get_report(self, kltime='2100-01-01'):
        self.__buy_sell_report(what='report', kltime=kltime)


class AnalystC(AnalystBase):
    def __init__(self, stockdata):
        AnalystBase.__init__(self, stockdata)


