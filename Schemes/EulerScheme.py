import numpy as np
from HestonMC.HestonModel import HestonModel
class EulerScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the EulerScheme instance. This class inherits from HestonModel.
        """
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt):
        """
        Update the volatility and asset price using the Euler method.
        param S: asset price
        param v: volatility
        param dt: time step
        return: updated asset price and volatility
        """

        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # 确保 Z1 和 Z2 的相关性为 rho

        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * np.sqrt(v) * np.sqrt(dt) * Z2

        v_next = max(v_next, 0)

        S_next = S * np.exp((self.r - 0.5 * v) * dt + np.sqrt(v) * np.sqrt(dt) * Z1)

        return S_next, v_next