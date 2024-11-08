import numpy as np
from HestonModel import HestonModel

class QEScheme(HestonModel):
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        """
        Initialize the QE (Quadratic Exponential) discretization scheme and inherit HestonModel parameters.
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
        Perform a single-step update using the QE discretization scheme.
        :param S: Current asset price
        :param v: Current volatility
        :param dt: Time step
        :return: Asset price S and volatility v at the next time step
        """
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # Ensure Z1 and Z2 are correlated with rho

        # Recalculate conditional mean m and conditional variance s2
        m = self.theta + (v - self.theta) * np.exp(-self.kappa * dt)
        s2 = v * (self.sigma ** 2) * np.exp(-self.kappa * dt) * (1 - np.exp(-self.kappa * dt)) / self.kappa

        psi = s2 / (m ** 2)  # Ratio to determine which method to use for v_next

        if psi < 1.5:
            # Update v_next using the non-central chi-squared distribution
            b = np.sqrt(2 / psi - 1)
            a = m / (1 + b ** 2)
            U = np.random.uniform()
            v_next = a * (b + np.sqrt(2 * U)) ** 2
        else:
            # Use Exponential distribution approach for v_next
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            U = np.random.uniform()
            v_next = m if U < p else np.log(1 - U) / (-beta)
        
        v_next = max(v_next, 0)  # Ensure non-negative volatility

        # Calculate the next asset price S_next using updated v_next
        S_next = S * np.exp((self.r - 0.5 * v_next) * dt + np.sqrt(v_next * dt) * Z1)

        return S_next, v_next
