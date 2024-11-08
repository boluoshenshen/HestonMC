import numpy as np
from HestonModel import HestonModel

class EulerScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the Euler discretization scheme and inherit HestonModel parameters.
        """
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt):
        """
        Perform a single-step update using the Euler discretization scheme.
        :param S: Current asset price
        :param v: Current volatility
        :param dt: Time step
        :return: Asset price S and volatility v at the next time step
        """
        # Generate two correlated normal random variables
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # Ensure Z1 and Z2 have correlation rho

        # Update volatility and asset price using the Euler method
        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * np.sqrt(v) * np.sqrt(dt) * Z2
        # Ensure non-negative volatility
        v_next = max(v_next, 0)

        S_next = S * np.exp((self.r - 0.5 * v) * dt + np.sqrt(v) * np.sqrt(dt) * Z1)

        return S_next, v_next
