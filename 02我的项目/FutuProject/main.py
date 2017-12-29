from StockData import StockData
from Analyst import Analyst
from .Operator import Operator


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
    sd = StockData("US.BABA")
    ay = Analyst(sd)
    op = Operator(0,100)

    # kp_array = ay.get_kp_array(todate='2017-02-14 10:40:00')
    # print(kp_array)
    # kp_array1 = ay.get_kp_array(todate='2017-02-15 10:13:00')
    # print(kp_array1)

    # ay.get_report()     # 根据最新资料报告买卖计划
    # ay.get_report(kltime='2017-02-15 10:13:00', cache=False)
    # print("-"*10)
    # ay.get_report(kltime='2017-03-08 09:45:00')
    trade = TradeLoopBack(sd, op, ay)
    trade.execute_trade()
    op.get_value(sd[-1].Close)      # 报告当前账户情况
    pass


