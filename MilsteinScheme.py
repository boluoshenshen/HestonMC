import numpy as np
from HestonModel import HestonModel

class MilsteinScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the Milstein discretization scheme and inherit HestonModel parameters.
        :param S0: Initial asset price
        :param v0: Initial volatility
        :param kappa: Mean reversion rate
        :param theta: Long-term mean
        :param sigma: Volatility of volatility
        :param rho: Correlation between asset price and volatility
        :param r: Risk-free interest rate
        """
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt):
        """
        Perform a single-step update using the Milstein discretization scheme.
        :param S: Current asset price
        :param v: Current volatility
        :param dt: Time step
        :return: Asset price S and volatility v at the next time step
        """
        # Generate two correlated normal random variables
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # Ensure Z1 and Z2 have correlation rho

        # Update volatility v using the Milstein method
        # Ensure volatility is non-negative
        v_sqrt = np.sqrt(v)
        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * v_sqrt * np.sqrt(dt) * Z2 \
                 + 0.25 * self.sigma**2 * dt * (Z2**2 - 1)  # Milstein correction term
        v_next = max(v_next, 0)  # Ensure non-negative volatility

        # Update asset price S
        S_next = S * np.exp((self.r - 0.5 * v) * dt + v_sqrt * np.sqrt(dt) * Z1 \
                            + 0.5 * v_sqrt * self.sigma * dt * (Z1**2 - 1))  # Milstein correction term

        return S_next, v_next
