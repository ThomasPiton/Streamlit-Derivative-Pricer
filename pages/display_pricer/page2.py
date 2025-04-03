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
position_type = st.selectbox("Position Type", ["LONG", "SHORT"])
S = st.number_input("Spot Price (S)", min_value=1.0, value=100.0, step=1.0)
K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.number_input("Time to Maturity (years)", min_value=0.01, value=1.0, step=0.1)
r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=3.0, step=0.1) / 100
q = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
repo = st.number_input("Repo Rate (%)", min_value=0.0, value=0.0, step=0.1) / 100
sigma = st.number_input("Volatility (%)", min_value=1.0, value=20.0, step=1.0) / 100

# Create Black-Scholes model instance
bsm = bsm(S=S, K=K, T=T, r=r, q=q, repo=repo, sigma=sigma, option_type=option_type, position_type=position_type)

visualizer = BlackScholesVisualizer(model=bsm)
spot_range = np.linspace(25, 175, 1000)
vol_range = np.linspace(0.01, 1, 1000)
time_range = np.linspace(0.01, 5, 1000)
rate_range = np.linspace(0.01, 1, 1000)
div_range = np.linspace(0.01, 1, 1000)
repo_range = np.linspace(0.01, 1, 1000)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs(
    [
        "d1","d2","Nd1","Nd2",
        "Payoff","Price",
        "Delta","Gamma","Vega","Theta","Rho",
        "Epsilon","Repo Sensi","Carry Rho", "Elasticity",
    ])

with tab1: 
    st.header("d1")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="d1",
        greek_func=bsm.param_d1,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="d1 vs Shifted Spot"))
    
with tab2: 
    st.header("d2")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="d2",
        greek_func=bsm.param_d2,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="d2 vs Shifted Spot"))
    
with tab3: 
    st.header("Nd1")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Nd1",
        greek_func=bsm.param_nd1,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Nd1 vs Shifted Spot"))
    
with tab4: 
    st.header("Nd2")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Nd2",
        greek_func=bsm.param_nd2,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Nd2 vs Shifted Spot"))

with tab5: 
    st.header("Payoff")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Payoff",
        greek_func=bsm.payoff,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Payoff vs Shifted Spot"))

with tab6: 
    st.header("Price")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Price",
        greek_func=bsm.price,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Price vs Shifted Spot"))
    
with tab7: 
    st.header("Delta")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Delta vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Delta vs Shifted Sigma"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="T",
        param_range=time_range,
        time_to_maturity=T,
        title="Delta vs Shifted Time"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="r",
        param_range=rate_range,
        interest_rate=r,
        title="Delta vs Shifted Rate"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="q",
        param_range=div_range,
        dividend=q,
        title="Delta vs Shifted Dividend"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Delta",
        greek_func=bsm.delta,
        param_name="repo",
        param_range=repo_range,
        repo=repo,
        title="Delta vs Shifted Repo"))
    
with tab8: 
    st.header("Gamma")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Gamma",
        greek_func=bsm.gamma,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Gamma vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Gamma",
        greek_func=bsm.gamma,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Gamma vs Shifted Sigma"))
    
with tab9: 
    st.header("Vega")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vega",
        greek_func=bsm.vega,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Vega vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vega",
        greek_func=bsm.vega,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Vega vs Shifted Sigma"))
    
with tab10: 
    st.header("Theta")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Theta",
        greek_func=bsm.theta,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Theta vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Theta",
        greek_func=bsm.theta,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Theta vs Shifted Sigma"))
    
with tab11: 
    st.header("Rho")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Rho",
        greek_func=bsm.rho,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Rho vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Rho",
        greek_func=bsm.rho,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Rho vs Shifted Sigma"))
    
with tab12: 
    st.header("Epsilon")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Epsilon",
        greek_func=bsm.epsilon,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Epsilon vs Shifted Spot"))

with tab13: 
    st.header("Repo Sensi")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Repo Sensitivity",
        greek_func=bsm.repo_sensitivity,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Repo Sensi vs Shifted Spot"))

with tab14: 
    st.header("Carry Rho")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Carry Rho",
        greek_func=bsm.carry_rho,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Carry Rho vs Shifted Spot"))
    
with tab15: 
    st.header("Elasticity")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Elasticity",
        greek_func=bsm.elasticity,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Elasticity vs Shifted Spot"))
    


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [
        "Vanna","Vomma","Veta","Vera", 
        "Charm","Speed","Zomma","Color",
        "Ultima","Dual Delta","Dual Gamma"
    ])

with tab1: 
    st.header("Vanna")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vanna",
        greek_func=bsm.vanna,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Vanna vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vanna",
        greek_func=bsm.vanna,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Vanna vs Shifted Sigma"))
    
with tab2: 
    st.header("Vomma")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vomma",
        greek_func=bsm.vomma,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Vomma vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vomma",
        greek_func=bsm.vomma,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Vomma vs Shifted Sigma"))
    
with tab3: 
    st.header("Veta")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Veta",
        greek_func=bsm.veta,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Veta vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Veta",
        greek_func=bsm.veta,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Veta vs Shifted Sigma"))
    
with tab4: 
    st.header("Vera")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vera",
        greek_func=bsm.vera,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Vera vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vera",
        greek_func=bsm.vera,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Vera vs Shifted Sigma"))
    
with tab5: 
    st.header("Charm")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Charm",
        greek_func=bsm.charm,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Charm vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Charm",
        greek_func=bsm.charm,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Charm vs Shifted Sigma"))
    
with tab6: 
    st.header("Speed")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Speed",
        greek_func=bsm.speed,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Speed vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Speed",
        greek_func=bsm.speed,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Speed vs Shifted Sigma"))
    
with tab7: 
    st.header("Zomma")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Zomma",
        greek_func=bsm.zomma,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Zomma vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Zomma",
        greek_func=bsm.zomma,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Zomma vs Shifted Sigma"))
    
with tab8: 
    st.header("Color")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Color",
        greek_func=bsm.color,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Color vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Color",
        greek_func=bsm.color,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Color vs Shifted Sigma"))
    
with tab9: 
    st.header("Ultima")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Ultima",
        greek_func=bsm.ultima,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Ultima vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Ultima",
        greek_func=bsm.ultima,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Ultima vs Shifted Sigma"))
    
with tab10: 
    st.header("Dual Delta")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Dual Delta",
        greek_func=bsm.dual_delta,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Dual Delta vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Dual Delta",
        greek_func=bsm.dual_delta,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Dual Delta vs Shifted Sigma"))
    
with tab11: 
    st.header("Dual Gamma")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Dual Gamma",
        greek_func=bsm.dual_gamma,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Dual Gamma vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Dual Gamma",
        greek_func=bsm.dual_gamma,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Dual Gamma vs Shifted Sigma"))