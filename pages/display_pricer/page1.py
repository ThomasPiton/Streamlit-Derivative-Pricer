import streamlit as st
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Callable
from pricer.bsm import BlackScholesModel as bsm
from pricer.bsm import BlackScholesVisualizer
from pricer.config import GREEK_LABELS, PARAM_LABELS

st.title("Black-Scholes Option Model Analyzer")
st.markdown("### Interactive visualization of option prices and Greeks")

# Common parameters in the sidebar
st.sidebar.header("Option Parameters")

option_type = st.selectbox("Option Type", ["CALL", "PUT"])
S = st.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0)
K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.1)
r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
sigma = st.number_input("Volatility (%)", min_value=1.0, value=20.0, step=1.0) / 100

# Create Black-Scholes model instance
bsm = bsm(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

# Calculate all Greeks
greeks = {
    'payoff': bsm.payoff(),
    'price': bsm.price(),
    'delta': bsm.delta(),
    'gamma': bsm.gamma(),
    'vega': bsm.vega() * 100,  # Scale back for display
    'theta': bsm.theta() * 365,  # Scale back for display (annual)
    'rho': bsm.rho() * 100,  # Scale back for display
    'vanna': bsm.vanna() * 100,  # Scale back for display
    'charm': bsm.charm() * 365,  # Scale back for display (annual)
    'speed': bsm.speed(),
    'zomma': bsm.zomma(),
    'color': bsm.color() * 365,  # Scale back for display (annual)
    'vomma': bsm.vomma(),
    'ultima': bsm.ultima()
}

# Display Greeks as metrics in sidebar
# st.sidebar.subheader("Option Greeks")
# for greek, value in greeks.items():
#     st.sidebar.metric(label=GREEK_LABELS[greek], value=round(value, 4))

# tab_names = [
#     "Payoff","Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Charm",
#     "Speed", "Zomma", "Color", "Vomma", "Ultima"
# ]
# tabs = st.tabs(tab_names)

# tab_formulas = {
#     "Payoff": "{Payoff (Call)} = \max(S - K, 0)",
#     "Price": "C = S N(d_1) - K e^{-rT} N(d_2)",
#     "Delta": "\\Delta = \\frac{\\partial C}{\\partial S}",
#     "Gamma": "\\Gamma = \\frac{\\partial^2 C}{\\partial S^2}",
#     "Vega": "\\nu = \\frac{\\partial C}{\\partial \\\sigma}",
#     "Theta": "\\Theta = \\frac{\\partial C}{\\partial T}",
#     "Rho": "\\rho = \\frac{\\partial C}{\\partial r}",
#     "Vanna": "Vanna = \\frac{\\partial^2 C}{\\partial S \\partial \\sigma}",
#     "Charm": "Charm = \\frac{\\partial \\Delta}{\\partial T}",
#     "Speed": "Speed = \\frac{\\partial^3 C}{\\partial S^3}",
#     "Zomma": "Zomma = \\frac{\\partial \\Gamma}{\\partial \\sigma}",
#     "Color": "Color = \\frac{\\partial \\Gamma}{\\partial T}",
#     "Vomma": "Vomma = \\frac{\\partial^2 C}{\\partial \\sigma^2}",
#     "Ultima": "Ultima = \\frac{\\partial Vomma}{\\partial \\sigma}"
# }

# visualizer = BlackScholesVisualizer(bsm)
# spot_range = np.linspace(50, 150, 100)

# for tab, name in zip(tabs, tab_names):
#     with tab:
#         st.header(name)
#         st.latex(tab_formulas[name])
#         visualizer.plot_greek_curve(
#             greek_name="Delta",
#             greek_func=bsm.delta,
#             param_name="S",
#             param_range=spot_range,
#             title="Option Delta vs Stock Price"
#         )


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
    [
        "Payoff & Price", 
        "Delta", 
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
        "Vanna",
        "Charm",
        "Speed",
        "Zomma",
        "Color",
        "Vomma",
        "Ultima"
    ]
)

with tab1: 
    st.header("Payoff & Price")
    
with tab2: 
    st.header("Delta")
with tab3: 
    st.header("Gamma")
    st.latex("")
with tab4: 
    st.header("Vega")
    st.latex("")
with tab5: 
    st.header("Theta")
    st.latex("")
with tab6: 
    st.header("Rho")
    st.latex("")
with tab7: 
    st.header("Vanna")
    st.latex("")
with tab8: 
    st.header("Charm")
    st.latex("")
with tab9: 
    st.header("Speed")
    st.latex("")
with tab10: 
    st.header("Zomma")
    st.latex("")
with tab11: 
    st.header("Color")
    st.latex("")
with tab12: 
    st.header("Vomma")
    st.latex("")
with tab13: 
    st.header("Ultima")
    st.latex("")