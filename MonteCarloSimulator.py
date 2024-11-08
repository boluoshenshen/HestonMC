import numpy as np


class MonteCarloSimulator:
    def __init__(self, model, scheme, num_paths, T, num_steps):
        self.model = model
        self.scheme = scheme
        self.num_paths = num_paths
        self.T = T
        self.num_steps = num_steps
        # Pre-generate Z1 and Z2 matrices
        self.Z1 = np.random.normal(size=(num_paths, num_steps))
        Z2_uncorrelated = np.random.normal(size=(num_paths, num_steps))
        self.Z2 = model.rho * self.Z1 + np.sqrt(1 - model.rho**2) * Z2_uncorrelated

    def generate_paths(self):
        S_paths = np.zeros((self.num_paths, self.num_steps + 1))
        v_paths = np.zeros((self.num_paths, self.num_steps + 1))

        for i in range(self.num_paths):
            S, v = self.model.S0, self.model.v0
            S_paths[i, 0], v_paths[i, 0] = S, v

            for j in range(1, self.num_steps + 1):
                # Pass pre-generated Z1 and Z2 values
                S, v = self.scheme.step(S, v, self.T / self.num_steps, self.Z1[i, j-1], self.Z2[i, j-1])
                S_paths[i, j] = S
                v_paths[i, j] = v

        return S_paths, v_paths

    def price_option(self, payoff_func):
        S_paths, _ = self.generate_paths()
        payoffs = np.maximum(payoff_func(S_paths[:, -1]), 0)
        discount_factor = np.exp(-self.model.r * self.T)
        return discount_factor * np.mean(payoffs)