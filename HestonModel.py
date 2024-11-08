import numpy as np

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize Heston model parameters.
        :param S0: Initial asset price
        :param v0: Initial volatility
        :param kappa: Mean reversion rate
        :param theta: Long-term mean
        :param sigma: Volatility of volatility
        :param rho: Correlation between asset price and volatility
        :param r: Risk-free interest rate
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
        Generate a path using the given discretization scheme.
        :param scheme: Discretization scheme instance (e.g., EulerScheme, QEScheme)
        :param T: Time to maturity
        :param num_steps: Number of time steps
        :return: Asset price and volatility paths
        """
        dt = T / num_steps  # Calculate each time step
        S, v = self.S0, self.v0  # Initial price and volatility
        S_path, v_path = [S], [v]  # Store price and volatility paths

        for _ in range(num_steps):
            S, v = scheme.step(S, v, dt)  # Update S and v using the discretization scheme
            S_path.append(S)
            v_path.append(v)

        return np.array(S_path), np.array(v_path)
