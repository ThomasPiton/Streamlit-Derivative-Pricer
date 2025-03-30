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
visualizer = BlackScholesVisualizer(model=bsm)
spot_range = np.linspace(25, 175, 1000)
vol_range = np.linspace(0.01, 1, 1000)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs(
    ["Payoff","Price","Delta","Gamma","Vega","Theta","Rho","Vanna","Charm","Speed","Zomma","Color","Vomma","Ultima"])

with tab1: 
    st.header("Payoff")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Payoff",
        greek_func=bsm.payoff,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Payoff vs Shifted Spot"))
    
with tab2: 
    st.header("Price")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Price",
        greek_func=bsm.price,
        param_name="S",
        strike_price=K,
        spot_price=S,
        param_range=spot_range,
        title="Price vs Shifted Spot"))
    
with tab3: 
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
    
with tab4: 
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
    
with tab5: 
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
    
with tab6: 
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
    
with tab7: 
    st.header("Rho")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Rho",
        greek_func=bsm.rho,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Price vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Rho",
        greek_func=bsm.rho,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Rho vs Shifted Sigma"))
    
with tab8: 
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
    
with tab9: 
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
    
with tab10: 
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
    
with tab11: 
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
    
with tab12: 
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
    
with tab13: 
    st.header("Vomma")
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vomma",
        greek_func=bsm.vomma,
        param_name="S",
        param_range=spot_range,
        strike_price=K,
        spot_price=S,
        title="Price vs Shifted Spot"))
    st.plotly_chart(visualizer.plot_greek_area_curve_custom(
        greek_name="Vomma",
        greek_func=bsm.vomma,
        param_name="sigma",
        param_range=vol_range,
        volatility=sigma,
        title="Vomma vs Shifted Sigma"))
    
with tab14: 
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
        title="Delta vs Shifted Sigma"))