from StockData import StockData
from Analyst import AnalystBase
from Operator import Operator


class TradeLoopBack:
    """
        交易回测系统
    """
    def __init__(self, stock_data, operator, analyst):
        self.__Stock_data = stock_data
        self.__operator = operator
        self.__analyst = analyst

    def execute_trade(self):
        """
            以时间驱动，完成交易回测
        """
        for x in self.__Stock_data:
            # print(x.KLTime)
            self.__operator.buy_strategy(self.__analyst, x)
            self.__operator.sell_strategy(self.__analyst, x)
        # return self.__operator.get_records()


if __name__ == '__main__':
    sd = StockData("US.NVDA", todate='2017-02-14 09:33:00')
    # ay = Analyst(sd)
    # ayB = AnalystB(sd, b_rate=-0.05, s_rate=0.10)
    # op = Operator(10000,100)

    # kp_array = ay.get_kp_array(todate='2017-02-14 10:40:00')
    # print(kp_array)
    # kp_array1 = ay.get_kp_array(todate='2017-02-15 10:13:00')
    # print(kp_array1)

    # ay.get_report()     # 根据最新资料报告买卖计划
    # ay.get_report(kltime='2017-02-15 10:13:00', cache=False)
    # print("-"*10)
    # ay.get_report(kltime='2017-03-08 09:45:00')

    # trade = TradeLoopBack(sd, op, ayB)
    # trade.execute_trade()
    # op.get_value(sd[-1].Close)      # 报告当前账户情况

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
    ayb = AnalystBase(sd)
    ayb.print_a('2017-02-10 11:32:00', '2017-02-10 12:32:00')

    pass


