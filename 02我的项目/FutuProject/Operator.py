from collections import namedtuple

class Operator:
    def __init__(self, money, share):
        self.__money = money
        self.__shares = share
        self.__his_value = []

    def buy_strategy(self, analyst, cur_time, cur_price):
        if analyst.can_buy(cur_time, cur_price) and (self.__money / cur_price) > 1:
            shares = int(self.__money / cur_price)
            self.__shares += shares
            self.__money -= shares * cur_price
            # print("%s buy %d shares at %.03f. money:%.03f" % (cur_time, shares, cur_price, self.__money))

    def sell_strategy(self, analyst, cur_time, cur_price):
        if analyst.can_sell(cur_time, cur_price) and self.__shares > 1:
            shares = self.__shares
            self.__money += shares * cur_price
            self.__shares = 0
            # print("%s sell %d shares at %.03f. money:%.03f" % (cur_time, shares, cur_price, self.__money))

    def get_value(self, cur_price):
        # print("current price:$%.03f Now have %d shares, $%.03f cash, total value: %.03f"
        #       % (cur_price, self.__shares, self.__money, (self.__shares*cur_price + self.__money)))
        return self.__shares*cur_price + self.__money

    def add_his_value(self, ind, cur_price):
        self.__his_value.append((ind, cur_price, (self.__shares*cur_price + self.__money)))
        pass