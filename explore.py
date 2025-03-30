from pricer.bsm import BlackScholesModel,BlackScholesVisualizer
import numpy as np

def run_vizualiser():
    bsm = BlackScholesModel(
        S=100,       # Current stock price
        K=100,       # Strike price (at the money)
        T=1.0,       # 1 year to expiration
        r=0.05,      # 5% risk-free rate
        sigma=0.20,  # 20% volatility
        option_type="call"
    )
    
    visualizer = BlackScholesVisualizer(bsm)
    
    # Example 1: Plot a single Greek curve (Delta vs Stock Price)
    vol_range = np.linspace(0.1, 0.5, 100)
    
    visualizer.plot_greek_curve(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="sigma",
        param_range=vol_range,
        title="Payoff vs Sigma")

if __name__ == '__main__':
    run_vizualiser()

