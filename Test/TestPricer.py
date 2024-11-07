import numpy as np
from HestonMC.Schemes.EulerScheme import EulerScheme
from HestonMC.MonteCarloSimulator import MonteCarloSimulator
from HestonMC.OptionPricer import OptionPricer

def main():
    S0 = 100         
    v0 = 0.04        
    kappa = 2.0      
    theta = 0.04     
    sigma = 0.3      
    rho = -0.7       
    r = 0.05         
    T = 1.0          
    K = 100          
    num_steps = 252  
    num_paths = 10000 

    euler_scheme = EulerScheme(S0, v0, kappa, theta, sigma, rho, r)
    mc_simulator = MonteCarloSimulator(euler_scheme, euler_scheme, num_paths, T, num_steps)
    option_pricer = OptionPricer(mc_simulator)
    call_price = option_pricer.european_call(K)
    print(f"European call option price: {call_price:.4f}")
    
if __name__ == "__main__":
    main()
