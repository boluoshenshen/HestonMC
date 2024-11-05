from MonteCarloSimulator import MonteCarloSimulator

class OptionPricer:
    def __init__(self, simulator):
        """
        初始化 OptionPricer
        :param simulator: MonteCarloSimulator 实例
        """
        self.simulator = simulator

    def european_call(self, K):
        """
        计算欧式看涨期权价格
        :param K: 执行价格（敲定价格）
        :return: 欧式看涨期权价格
        """
        call_payoff = lambda S: max(S - K, 0)
        return self.simulator.price_option(call_payoff)

    def european_put(self, K):
        """
        计算欧式看跌期权价格
        :param K: 执行价格（敲定价格）
        :return: 欧式看跌期权价格
        """
        put_payoff = lambda S: max(K - S, 0)
        return self.simulator.price_option(put_payoff)
