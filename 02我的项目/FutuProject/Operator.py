

class Operator:
    def __init__(self, money, share):
        self.__money = money
        self.__shares = share

    def buy_strategy(self, analyst, current):
        if analyst.can_buy(current[0]) and (self.__money / current[1].Close) > 1:
            shares = int(self.__money / current[1].Close)
            self.__shares += shares
            self.__money -= shares * current[1].Close
            print("%s buy %d shares at %.03f. money:%.03f" % (current[0], shares, current[1].Close, self.__money))

    def sell_strategy(self, analyst, current):
        if analyst.can_sell(current[0]) and self.__shares > 1:
            shares = self.__shares
            self.__money += shares * current[1].Close
            self.__shares = 0
            print("%s sell %d shares at %.03f. money:%.03f" % (current[0], shares, current[1].Close, self.__money))

    def get_value(self, cur_price):
        print("Now have %d shares, $%.03f cash, total value: %.03f"
              % (self.__shares, self.__money, (self.__shares*cur_price + self.__money)))


