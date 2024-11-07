import numpy as np
from HestonMC.HestonModel import HestonModel

class MilsteinScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the MilsteinScheme instance. This class inherits from HestonModel.
        """
        super().__init__(S0, v0, kappa, theta, sigma, rho, r)

    def step(self, S, v, dt):
        """
        Update the volatility and asset price using the Milstein method.
        param S: asset price
        param v: volatility
        param dt: time step
        return: updated asset price and volatility
        """
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  

        
        v_sqrt = np.sqrt(v)
        v_next = v + self.kappa * (self.theta - v) * dt + self.sigma * v_sqrt * np.sqrt(dt) * Z2 \
                 + 0.25 * self.sigma**2 * dt * (Z2**2 - 1) 
        v_next = max(v_next, 0)  # make sure v_next is non-negative

        S_next = S * np.exp((self.r - 0.5 * v) * dt + v_sqrt * np.sqrt(dt) * Z1 \
                            + 0.5 * v_sqrt * self.sigma * dt * (Z1**2 - 1))  
        return S_next, v_next
