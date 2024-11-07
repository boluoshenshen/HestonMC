from MonteCarloSimulator import MonteCarloSimulator
import numpy as np
class OptionPricer:
    def __init__(self, simulator):
        """
        Initialize an OptionPricer instance.
        """
        self.simulator = simulator

    def european_call(self, K):
        """
        Calculate the price of a European call option.
        :param K: strike price
        :return: European call option price
        """
        call_payoff = lambda S: np.maximum(S - K, 0)
        return self.simulator.price_option(call_payoff)

    def european_put(self, K):
        """
        Calculate the price of a European put option.
        :param K: strike price
        :return: European put option price
        """
        put_payoff = lambda S: np.maximum(K - S, 0)
        return self.simulator.price_option(put_payoff)
