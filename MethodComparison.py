import numpy as np
from MonteCarloSimulator import MonteCarloSimulator
from OptionPricer import OptionPricer
from HestonModel import HestonModel

class MethodComparison:
    def __init__(self, heston_model, T, K, true_price):
        """
        Initialize a MethodComparison instance.
        :param heston_model: Instance of the HestonModel
        :param T: Time to maturity
        :param K: Strike price
        :param true_price: Theoretical true price of the option
        """
        self.heston_model = heston_model
        self.T = T
        self.K = K
        self.true_price = true_price

    def accuracy(self, scheme, num_paths, num_steps):
        """
        Calculate the accuracy of different pricing methods.
        :param scheme: Discretization scheme instance
        :param num_paths: Number of simulated paths
        :param num_steps: Number of time steps
        :return: Absolute difference between estimated price and true price
        """
        simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
        option_pricer = OptionPricer(simulator)
        estimated_price = option_pricer.european_call(self.K)
        return abs(estimated_price - self.true_price)

    def convergence(self, scheme, num_paths, step_sizes):
        """
        Calculate the convergence of different methods.
        :param scheme: Discretization scheme instance
        :param num_paths: Number of simulated paths
        :param step_sizes: List of different time steps for convergence testing
        :return: List of errors for each time step
        """
        errors = []
        for num_steps in step_sizes:
            error = self.accuracy(scheme, num_paths, num_steps)
            errors.append(error)
        return errors

    def stability(self, scheme, num_paths, num_steps, num_trials=10):
        """
        Calculate the stability of different methods.
        :param scheme: Discretization scheme instance
        :param num_paths: Number of simulated paths
        :param num_steps: Number of time steps
        :param num_trials: Number of trials for stability testing
        :return: Standard deviation of option prices across trials
        """
        prices = []
        for _ in range(num_trials):
            simulator = MonteCarloSimulator(self.heston_model, scheme, num_paths, self.T, num_steps)
            option_pricer = OptionPricer(simulator)
            price = option_pricer.european_call(self.K)
            prices.append(price)
        return np.std(prices)
