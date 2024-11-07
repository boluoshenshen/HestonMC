import numpy as np
from HestonModel import HestonModel

class MonteCarloSimulator:
    def __init__(self, model, scheme, num_paths, T, num_steps):
        """
        Initialize a MonteCarloSimulator instance.
        :param model: HestonModel instance
        :param scheme: discretization scheme instance
        :param num_paths: number of paths
        :param T: time to maturity
        """
        self.model = model
        self.scheme = scheme
        self.num_paths = num_paths
        self.T = T
        self.num_steps = num_steps

    def generate_paths(self):
        """
        Generate asset price and volatility paths.
        :return: asset price paths and volatility paths
        """
        S_paths = np.zeros((self.num_paths, self.num_steps + 1))
        v_paths = np.zeros((self.num_paths, self.num_steps + 1))

        for i in range(self.num_paths):
            S, v = self.model.S0, self.model.v0  # 每条路径的初始值
            S_paths[i, 0], v_paths[i, 0] = S, v

            for j in range(1, self.num_steps + 1):
                S, v = self.scheme.step(S, v, self.T / self.num_steps)
                S_paths[i, j] = S
                v_paths[i, j] = v

        return S_paths, v_paths

    def price_option(self, payoff_func):
        """
        Calculate the price of an option.
        :param payoff_func: payoff function
        :return: option price
        """
        S_paths, _ = self.generate_paths()
        payoffs = np.maximum(payoff_func(S_paths[:, -1]), 0)
        discount_factor = np.exp(-self.model.r * self.T)
        return discount_factor * np.mean(payoffs)
