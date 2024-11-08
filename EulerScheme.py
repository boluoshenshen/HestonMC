import numpy as np
from HestonModel import HestonModel
class EulerScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt, Z1, Z2):
        """
        Update the volatility and asset price using the Euler method.
        param S: asset price
        param v: volatility
        param dt: time step
        param Z1, Z2: pre-generated random variables
        return: updated asset price and volatility
        """
        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * np.sqrt(v) * np.sqrt(dt) * Z2
        v_next = max(v_next, 0)
        S_next = S * np.exp((self.r - 0.5 * v) * dt + np.sqrt(v) * np.sqrt(dt) * Z1)
        return S_next, v_next