import numpy as np

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the HestonModel instance.
        :param S0: initial price
        :param v0: initial variance
        :param kappa: mean reversion speed
        :param theta: long-term mean
        :param sigma: volatility of volatility
        :param rho: correlation between asset price and volatility
        :param r: risk-free rate
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def simulate_path(self, scheme, T, num_steps):
        """
        Simulate a path of asset prices and volatilities.
        :param scheme: discretization scheme instance (e.g., EulerScheme or QEScheme)
        :param T: time to maturity
        :param num_steps: number of time steps
        return: asset price path and volatility path
        """
        dt = T / num_steps  
        S, v = self.S0, self.v0 
        S_path, v_path = [S], [v]  # list of paths

        for _ in range(num_steps):
            S, v = scheme.step(S, v, dt)  
            S_path.append(S)
            v_path.append(v)

        return np.array(S_path), np.array(v_path)
