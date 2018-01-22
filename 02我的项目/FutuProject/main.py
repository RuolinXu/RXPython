from StockData import StockData
from Analyst import AnalystB
from Operator import Operator
from datetime import datetime

G_cache = {}


class TradeLoopBack:
    """
        交易回测系统
    """
    def __init__(self, stock_data, operator, analyst):
        self.stock_data = stock_data
        self.__operator = operator
        self.__analyst = analyst

    def execute_trade(self):
        """
            以时间驱动，完成交易回测
        """
        start_time = datetime.now()
        for x in self.stock_data:
            # print(x.KLTime)
            cur_time, cur_price = x.KLTime, x.Close
            self.__operator.buy_strategy(self.__analyst, cur_time, cur_price)
            self.__operator.sell_strategy(self.__analyst, cur_time, cur_price)
        end_time = datetime.now()
        # self.__operator.get_value(self.stock_data[-1].Close)
        print("耗时%d秒" % (end_time - start_time).seconds)


def stock_days(params):
    days = params[0]
    b_rate = params[1]*-1
    s_rate = params[2]
    print(params)
    print(len(G_cache))
    sd = StockData('US.BABA')
    ay = AnalystB(G_cache, sd, days, b_rate, s_rate)
    op = Operator(10000, 0)
    trade = TradeLoopBack(sd, op, ay)
    trade.execute_trade()
    return op.get_value(sd[-1].Close) * -1



if __name__ == '__main__':
    # sd = StockData("US.BABA")
    # print(type(sd))
    # ay = Analyst(sd)
    # ayB = AnalystB(G_cache, sd,5, b_rate=-0.04, s_rate=0.137)
    # op = Operator(10000, 0)

    # trade = TradeLoopBack(sd, op, ayB)
    # trade.execute_trade()
    # op.get_value(sd[-1].Close)      # 报告当前账户情况

    import scipy.optimize as sco
    opt_global = sco.brute(stock_days, ((5, 10, 1), (0.03, 0.1, 0.01), (0.05, 0.15, 0.01)))
    print(opt_global)


    # kp_array = ay.get_kp_array(todate='2017-02-14 10:40:00')
    # print(kp_array)
    # kp_array1 = ay.get_kp_array(todate='2017-02-15 10:13:00')
    # print(kp_array1)

    # ay.get_report()     # 根据最新资料报告买卖计划
    # ay.get_report(kltime='2017-02-15 10:13:00', cache=False)
    # print("-"*10)
    # ay.get_report(kltime='2017-03-08 09:45:00')



    # kp = ay.get_kp_array(rate=0.05)
    # print(kp)
    # ay.get_keypoint_report(kp)

    # print('测试get_kpm_array 2017-03-15 10:13:00')
    # kp = ayB.get_kpm_array(todate='2017-03-15 10:13:00')
    # for x in kp:
    #     print(x)
    #
    # print('测试get_kpm_array 2017-03-17 10:13:00')
    # kp2 = ayB.get_kpm_array(todate='2017-03-17 10:13:00')
    # print(len(kp2))
    # for x in kp2:
    #     print(x)

    # ayB.get_report()
    # ayb = AnalystBase(sd)
    # ayb.print_a('2017-02-10 11:32:00', '2017-02-10 12:32:00')

    pass


