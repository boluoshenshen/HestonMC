import numpy as np
from HestonModel import HestonModel
from EulerScheme import EulerScheme
from MonteCarloSimulator import MonteCarloSimulator
from OptionPricer import OptionPricer

def main():
    # Heston model parameters
    S0 = 100         # Initial asset price
    v0 = 0.04        # Initial volatility
    kappa = 2.0      # Speed of mean reversion
    theta = 0.04     # Long-term volatility level
    sigma = 0.3      # Volatility of volatility
    rho = -0.7       # Correlation between asset price and volatility
    r = 0.05         # Risk-free interest rate
    T = 1.0          # Time to maturity
    K = 100          # Strike price
    num_steps = 252  # Number of time steps
    num_paths = 10000  # Number of Monte Carlo paths

    # Initialize Heston model and Euler scheme
    heston_model = HestonModel(S0, v0, kappa, theta, sigma, rho, r)
    euler_scheme = EulerScheme(S0, v0, kappa, theta, sigma, rho, r)
    
    # Monte Carlo simulator with pre-generated random variables
    mc_simulator = MonteCarloSimulator(heston_model, euler_scheme, num_paths, T, num_steps)
    option_pricer = OptionPricer(mc_simulator)
    
    # Calculate and print the European call option price
    call_price = option_pricer.european_call(K)
    print(f"European call option price: {call_price:.4f}")

if __name__ == "__main__":
    main()
