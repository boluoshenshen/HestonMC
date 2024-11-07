import numpy as np
from HestonMC.HestonModel import HestonModel
from HestonMC.Schemes.EulerScheme import EulerScheme
from HestonMC.Schemes.MilsteinScheme import MilsteinScheme
from HestonMC.MethodComparison import MethodComparison



# Model parameters
S0 = 100        
v0 = 0.04       
kappa = 1.5  
theta = 0.04  
sigma = 0.3  
rho = -0.7  
r = 0.03   
T = 1.0   
K = 100   
true_price = 10  # Assumed true option price for comparison

heston_model = HestonModel(S0, v0, kappa, theta, sigma, rho, r)

# Instantiate each scheme
euler_scheme = EulerScheme(S0, v0, kappa, theta, sigma, rho, r)
milstein_scheme = MilsteinScheme(S0, v0, kappa, theta, sigma, rho, r)

# Create MethodComparison instance
comparison = MethodComparison(heston_model, T, K, true_price)

# Parameters for the comparison
num_paths = 1000      # Number of Monte Carlo paths
step_sizes = [10, 50, 100]  # Different step sizes for convergence testing

# Accuracy comparison
euler_accuracy = comparison.accuracy(euler_scheme, num_paths, num_steps=100)
milstein_accuracy = comparison.accuracy(milstein_scheme, num_paths, num_steps=100)
print("Euler Scheme Accuracy:", euler_accuracy)
print("Milstein Scheme Accuracy:", milstein_accuracy)

# Convergence comparison
euler_convergence = comparison.convergence(euler_scheme, num_paths, step_sizes)
milstein_convergence = comparison.convergence(milstein_scheme, num_paths, step_sizes)
print("Euler Scheme Convergence:", euler_convergence)
print("Milstein Scheme Convergence:", milstein_convergence)

# Stability comparison
euler_stability = comparison.stability(euler_scheme, num_paths, num_steps=100, num_trials=10)
milstein_stability = comparison.stability(milstein_scheme, num_paths, num_steps=100, num_trials=10)
print("Euler Scheme Stability:", euler_stability)
print("Milstein Scheme Stability:", milstein_stability)
