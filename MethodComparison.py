import numpy as np
from HestonMC.MonteCarloSimulator import MonteCarloSimulator
from HestonMC.OptionPricer import OptionPricer


class MethodComparison:
    def __init__(self, heston_model, T, K, true_price):
        """
        Initialize a MethodComparison instance.
        :param heston_model: HestonModel instance
        :param T: time to maturity
        :param K: strike price
        """
        self.heston_model = heston_model
        self.T = T
        self.K = K
        self.true_price = true_price

    def accuracy(self, scheme, num_paths, num_steps):
        """
        Calculate the accuracy of different methods.
        :param scheme: discretization scheme instance
        :param num_paths: number of paths
        :param num_steps: number of time steps
        :return: absolute difference between estimated price and true price
        """
        simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
        option_pricer = OptionPricer(simulator)
        estimated_price = option_pricer.european_call(self.K)
        return abs(estimated_price - self.true_price)

    def convergence(self, scheme, num_paths, step_sizes):
        """
        Calculate the convergence of different methods.
        :param scheme: discretization scheme instance
        :param num_paths: number of paths
        :param step_sizes: list of time steps
        :return: list of errors
        """
        errors = []
        for num_steps in step_sizes:
            error = self.accuracy(scheme, num_paths, num_steps)
            errors.append(error)
        return errors

    def stability(self, scheme, num_paths, num_steps, num_trials=10):
        """
        Calculate the stability of different methods.
        :param scheme: discretization scheme instance
        :param num_paths: number of paths
        :param num_steps: number of time steps
        :param num_trials: number of trials
        return: standard deviation of option prices
        """
        prices = []
        for _ in range(num_trials):
            simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
            option_pricer = OptionPricer(simulator)
            price = option_pricer.european_call(self.K)
            prices.append(price)
        return np.std(prices)
