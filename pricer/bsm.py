import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Callable
from pricer.config import PARAM_LABELS, GREEK_LABELS
import plotly.graph_objs as go 
import plotly.express as px
import pandas as pd

class BlackScholesModel:
    """
    Black-Scholes model for pricing European options and computing Greeks up to third order.
    
    Attributes:
        S (float): Spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'call' or 'put' to specify the option type.
    
    Methods:
        price(): Computes the option price.
        delta(): Computes the first-order sensitivity to spot price.
        gamma(): Computes the second-order sensitivity to spot price.
        vega(): Computes the sensitivity to volatility.
        theta(): Computes the sensitivity to time decay.
        rho(): Computes the sensitivity to interest rate changes.
        charm(): Computes the rate of change of delta over time.
        speed(): Computes the rate of change of gamma with respect to spot price.
        zomma(): Computes the rate of change of gamma with respect to volatility.
        color(): Computes the rate of change of gamma over time.
        ultima(): Computes the rate of change of vomma with respect to volatility.
        vomma(): Computes the rate of change of vega with respect to volatility.
        vanna(): Computes the sensitivity of delta to volatility (or vega to spot price).
        dual_delta(): Computes the sensitivity to changes in strike price.
        dual_gamma(): Computes the second-order sensitivity to strike price.
        compute_greek_curve(): Compute a Greek value across a range of parameter values.
        plot_greek_curves(): Plot multiple Greek curves with various parameter shifts.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call'):
        """Initialize the Black-Scholes model with option parameters."""
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()
        
        # Validate option type
        if self.option_type not in ['call', 'put']:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        # Pre-compute d1 and d2 to avoid redundant calculations
        self._update_d_values()
    
    def _update_d_values(self):
        """Update d1 and d2 values based on current parameters."""
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
    
    def payoff(self) -> float:
        """Calculate the option payoff."""
        if self.option_type == 'call':
            return max(self.S - self.K, 0)
        else:  # put option
            return max(self.K - self.S, 0)
    
    def price(self) -> float:
        """Calculate the option price."""
        if self.option_type == 'call':
            return self.S * si.norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2)
        else:  # put option
            return self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2) - self.S * si.norm.cdf(-self.d1)
    
    def delta(self) -> float:
        """Calculate the option delta (first derivative with respect to spot price)."""
        if self.option_type == 'call':
            return si.norm.cdf(self.d1)
        else:  # put option
            return si.norm.cdf(self.d1) - 1
    
    def gamma(self) -> float:
        """Calculate the option gamma (second derivative with respect to spot price)."""
        return si.norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """Calculate the option vega (sensitivity to volatility)."""
        return self.S * si.norm.pdf(self.d1) * np.sqrt(self.T) / 100  # Scaled by 100 for better readability
    
    def theta(self) -> float:
        """Calculate the option theta (sensitivity to time decay)."""
        term1 = -(self.S * si.norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if self.option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2)
        else:  # put option
            term2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2)
            
        return (term1 + term2) / 365  # Daily theta (scaled to trading days)
    
    def rho(self) -> float:
        """Calculate the option rho (sensitivity to interest rate changes)."""
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2) / 100  # Scaled by 100
        else:  # put option
            return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2) / 100  # Scaled by 100
    
    def charm(self) -> float:
        """Calculate the option charm (rate of change of delta over time)."""
        return -si.norm.pdf(self.d1) * (2 * (self.r - self.sigma**2 / 2) * self.T - self.d2 * self.sigma * np.sqrt(self.T)) / (2 * self.S * self.sigma * self.T * np.sqrt(self.T)) / 365  # Daily charm
    
    def speed(self) -> float:
        """Calculate the option speed (rate of change of gamma with respect to spot price)."""
        return -self.gamma() / self.S * (self.d1 / (self.sigma * np.sqrt(self.T)) + 1)
    
    def zomma(self) -> float:
        """Calculate the option zomma (rate of change of gamma with respect to volatility)."""
        return self.gamma() * (self.d1 * self.d2 - 1) / self.sigma
    
    def color(self) -> float:
        """Calculate the option color (rate of change of gamma over time)."""
        return (-si.norm.pdf(self.d1) / (2 * self.S * self.T * np.sqrt(self.T)) * (2 * self.r * self.T + 1 + self.d1 * self.d2)) / 365  # Daily color
    
    def ultima(self) -> float:
        """Calculate the option ultima (rate of change of vomma with respect to volatility)."""
        return -self.vomma() * (self.d1 * self.d2 + 1) / self.sigma
    
    def vomma(self) -> float:
        """Calculate the option vomma (rate of change of vega with respect to volatility)."""
        return self.vega() * 100 * (self.d1 * self.d2)  # Rescaled because vega is already scaled
    
    def vanna(self) -> float:
        """
        Calculate the option vanna (cross partial derivative of delta with respect to volatility, 
        or equivalently, the partial derivative of vega with respect to spot price).
        
        Vanna measures how the delta of an option changes with respect to a change in volatility,
        or equivalently, how vega changes with respect to a change in the underlying price.
        
        Returns:
            float: The vanna value.
        """
        # Vanna is the same for both call and put options
        return -si.norm.pdf(self.d1) * self.d2 / self.sigma
    
    def dual_delta(self) -> float:
        """Calculate the option dual delta (sensitivity to changes in strike price)."""
        if self.option_type == 'call':
            return -np.exp(-self.r * self.T) * si.norm.cdf(self.d2)
        else:  # put option
            return np.exp(-self.r * self.T) * si.norm.cdf(-self.d2)
    
    def dual_gamma(self) -> float:
        """Calculate the option dual gamma (second-order sensitivity to strike price)."""
        return np.exp(-self.r * self.T) * si.norm.pdf(self.d2) / (self.K * self.sigma * np.sqrt(self.T))
    
    def get_params(self) -> Dict[str, float]:
        """Get all model parameters as a dictionary."""
        return {
            'S': self.S,
            'K': self.K,
            'T': self.T,
            'r': self.r,
            'sigma': self.sigma
        }
    


class BlackScholesVisualizer:
    """
    Visualizer for Black-Scholes model results.
    
    Responsabilité: Créer des visualisations pour les résultats des calculs du modèle.
    """
    def __init__(self, model):
        """Initialize with a Black-Scholes model instance."""
        self.model = model
        import plotly.graph_objs as go
        import plotly.express as px
        self.go = go
        self.px = px
    
    def compute_greek_curve(
        self, 
        greek_func: Callable, 
        param_name: str, 
        param_range: List[float]
    ) -> Dict[str, List[float]]:
        """
        Compute a Greek value across a range of parameter values.
        
        Args:
            greek_func: Method of BlackScholesModel that calculates the desired Greek
            param_name: Name of the parameter to vary ('S', 'K', 'T', 'r', or 'sigma')
            param_range: List of parameter values to compute the Greek for
            
        Returns:
            Dictionary with parameter values and corresponding Greek values
        """
        original_value = getattr(self.model, param_name)
        greek_values = []
        
        for param_value in param_range:
            # Temporarily update the parameter
            setattr(self.model, param_name, param_value)
            self.model._update_d_values()
            
            # Calculate and store the Greek value
            greek_values.append(greek_func())
        
        # Restore the original parameter value
        setattr(self.model, param_name, original_value)
        self.model._update_d_values()
        
        return {
            'param_values': param_range,
            'greek_values': greek_values
        }
    
    def plot_greek_curve(
        self, 
        greek_name: str, 
        greek_func: Callable, 
        param_name: str, 
        param_range: List[float], 
        title: str = None,
        xlabel: str = None,
        ylabel: str = None
    ) -> go.Figure:
        """
        Plot a Greek value across a range of parameter values.
        
        Args:
            greek_name: Name of the Greek (for labeling)
            greek_func: Function that calculates the Greek
            param_name: Name of the parameter to vary
            param_range: List of parameter values
            title, xlabel, ylabel: Optional plot labels
        
        Returns:
            Plotly figure with the plotted curve
        """
        data = self.compute_greek_curve(greek_func, param_name, param_range)
        
        fig = self.go.Figure()
        fig.add_trace(
            self.go.Scatter(
                x=data['param_values'], 
                y=data['greek_values'], 
                mode='lines', 
                line=dict(color='blue', width=2),
                name=greek_name
            )
        )
        
        # Set labels and title
        fig.update_layout(
            title=title or f'{greek_name} vs {PARAM_LABELS.get(param_name, param_name)}',
            xaxis_title=xlabel or PARAM_LABELS.get(param_name, param_name),
            yaxis_title=ylabel or greek_name,
            template="plotly_white"
        )
        
        return fig
    
    def plot_greek_area_curve(self, greek_name, greek_func, param_name, param_range, title=None, xlabel=None, ylabel=None):
        """
        Plot a Greek value across a range of parameter values using an area chart.
        """
        data = self.compute_greek_curve(greek_func, param_name, param_range)
        df = pd.DataFrame({
            'param_values': data['param_values'],
            'greek_values': data['greek_values']
        })
        
        fig = self.px.area(df, x='param_values', y='greek_values', title=title or f'{greek_name} vs {PARAM_LABELS.get(param_name, param_name)}')
        fig.update_layout(
            xaxis_title=xlabel or PARAM_LABELS.get(param_name, param_name),
            yaxis_title=ylabel or greek_name,
            template="plotly_white"
        )
        
        return fig
    
    def plot_greek_area_curve_custom(self, greek_name, greek_func, param_name, param_range, 
                                 strike_price=None, spot_price=None, time_to_maturity=None, 
                                 interest_rate=None, volatility=None, title=None, xlabel=None, ylabel=None):
        """
        Plot a Greek value across a range of parameter values using an area chart and a line curve with optional vertical lines.
        """
        data = self.compute_greek_curve(greek_func, param_name, param_range)
        df = pd.DataFrame({
            'param_values': data['param_values'],
            'greek_values': data['greek_values']
        })

        # Split data into positive and negative parts
        df_positive = df.copy()
        df_negative = df.copy()

        df_positive.loc[df_positive['greek_values'] < 0, 'greek_values'] = 0
        df_negative.loc[df_negative['greek_values'] > 0, 'greek_values'] = 0

        # Create figure
        fig = go.Figure()

        # Add positive area
        fig.add_trace(go.Scatter(
            x=df_positive['param_values'],
            y=df_positive['greek_values'],
            fill='tozeroy',
            mode='none',
            name='Positive Area',
            fillcolor='rgba(0, 106, 78, 0.5)'  # Semi-transparent green
        ))

        # Add negative area
        fig.add_trace(go.Scatter(
            x=df_negative['param_values'],
            y=df_negative['greek_values'],
            fill='tozeroy',
            mode='none',
            name='Negative Area',
            fillcolor='rgba(230, 0, 40, 0.5)'  # Semi-transparent red
        ))

        # Dictionary of vertical lines with their respective colors
        vertical_lines = {
            'strike_price': ('yellow', strike_price),
            'spot_price': ('cyan', spot_price),
            'time_to_maturity': ('orange', time_to_maturity),
            'interest_rate': ('blue', interest_rate),
            'volatility': ('purple', volatility)
        }

        # Add optional vertical lines
        for label, (color, value) in vertical_lines.items():
            if value is not None:
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=value,
                        x1=value,
                        y0=min(df['greek_values']),
                        y1=max(df['greek_values']),
                        line=dict(color=color, width=2, dash="dot"),
                    )
                )
                # Add dummy trace for legend entry
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],  # Invisible data points
                    mode='lines',
                    line=dict(color=color, width=2, dash="dot"),
                    name=label  # Use the label for the legend
                ))

        # Layout settings
        fig.update_layout(
            title=title or f'{greek_name} vs {PARAM_LABELS.get(param_name, param_name)}',
            xaxis_title=xlabel or PARAM_LABELS.get(param_name, param_name),
            yaxis_title=ylabel or greek_name,
            template="plotly_dark",  # Dark theme for contrast
            legend=dict(
                orientation="h",  # Set horizontal orientation
                x=0.5,  # Center horizontally
                xanchor="center",  # Anchor the legend at the center
                y=-0.2  # Position the legend below the plot (adjust as necessary)
            )
        )

        return fig
            
    def compute_greek_curve_dynamic(
        self, 
        greek: str, 
        x_param: str, 
        range_pct: float = 0.2, 
        points: int = 100, 
        base_shift: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the curve of a specific Greek with respect to a parameter.
        
        Args:
            greek (str): The Greek to compute ('delta', 'gamma', etc.).
            x_param (str): The parameter to vary ('S', 'K', 'T', 'r', 'sigma').
            range_pct (float): The range to vary the parameter by, as a percentage.
            points (int): Number of points in the curve.
            base_shift (float): Base shift to apply to all parameters.
            
        Returns:
            Tuple of (x_values, greek_values)
        """
        # Validate inputs
        if greek not in GREEK_LABELS:
            raise ValueError(f"Unknown Greek: {greek}. Available Greeks: {', '.join(GREEK_LABELS.keys())}")
        
        if x_param not in PARAM_LABELS:
            raise ValueError(f"Unknown parameter: {x_param}. Available parameters: {', '.join(PARAM_LABELS.keys())}")
        
        # Get current parameter value
        base_value = getattr(self.model, x_param)
        
        # Create range of parameter values
        min_value = base_value * (1 - range_pct) + base_shift
        max_value = base_value * (1 + range_pct) + base_shift
        
        # Ensure positive values for certain parameters
        if x_param in ['S', 'K', 'sigma', 'T'] and min_value <= 0:
            min_value = base_value * 0.01  # Small positive value
            
        param_values = np.linspace(min_value, max_value, points)
        greek_values = []
        
        # Get the method to compute the Greek
        greek_method = getattr(self.model, greek.lower())
        
        # Compute Greek for each parameter value
        for param_value in param_values:
            # Save original value
            original_value = getattr(self.model, x_param)
            
            # Set new value
            setattr(self.model, x_param, param_value)
            self.model._update_d_values()
            
            # Calculate Greek
            greek_values.append(greek_method())
            
            # Restore original value
            setattr(self.model, x_param, original_value)
            self.model._update_d_values()
            
        return param_values, np.array(greek_values)
    
    def plot_greek_curves_dynamic(
        self, 
        greek: str, 
        x_param: str, 
        shifts: List[float] = None,
        shift_param: str = None,
        range_pct: float = 0.2, 
        points: int = 100,
        title: str = None
    ) -> go.Figure:
        """
        Plot curves for a specific Greek with multiple parameter shifts.
        
        Args:
            greek (str): The Greek to compute ('delta', 'gamma', etc.).
            x_param (str): The parameter for the x-axis ('S', 'K', 'T', 'r', 'sigma').
            shifts (List[float], optional): List of shifts to apply to shift_param.
            shift_param (str, optional): Parameter to shift for multiple curves.
            range_pct (float): The range to vary the x_param by, as a percentage.
            points (int): Number of points in the curve.
            title (str, optional): Custom title for the plot.
            
        Returns:
            Plotly figure with the plotted curves
        """
        if shifts is None:
            shifts = [0.0]  # Default to no shift
            
        if shift_param is None:
            shift_param = x_param  # Default to shifting the x-axis parameter
            
        # Create the plot
        fig = self.go.Figure()
        
        # Store original values to restore later
        original_shift_value = getattr(self.model, shift_param)
        original_x_value = getattr(self.model, x_param)
        
        # For each shift, compute and plot a curve
        for shift in shifts:
            # If not shifting the x-axis parameter
            if shift_param != x_param:
                # Apply shift to the shift parameter
                shifted_value = original_shift_value * (1 + shift)
                setattr(self.model, shift_param, shifted_value)
                self.model._update_d_values()
                
                # Compute curve for the x parameter
                x_values = np.linspace(
                    original_x_value * (1 - range_pct), 
                    original_x_value * (1 + range_pct), 
                    points
                )
                greek_values = []
                
                for x_value in x_values:
                    setattr(self.model, x_param, x_value)
                    self.model._update_d_values()
                    greek_values.append(getattr(self.model, greek.lower())())
                
            else:
                # If shifting the x-axis parameter itself, adjust the range
                x_values = np.linspace(
                    original_x_value * (1 - range_pct) * (1 + shift), 
                    original_x_value * (1 + range_pct) * (1 + shift), 
                    points
                )
                greek_values = []
                
                for x_value in x_values:
                    setattr(self.model, x_param, x_value)
                    self.model._update_d_values()
                    greek_values.append(getattr(self.model, greek.lower())())
            
            # Generate label for the curve
            if shift == 0:
                label = f"Base {PARAM_LABELS[shift_param]}"
            else:
                shift_pct = shift * 100
                label = f"{PARAM_LABELS[shift_param]} {'+' if shift > 0 else ''}{shift_pct:.0f}%"
                
            # Plot the curve
            fig.add_trace(
                self.go.Scatter(
                    x=x_values,
                    y=greek_values,
                    mode='lines',
                    name=label
                )
            )
            
        # Restore original values
        setattr(self.model, shift_param, original_shift_value)
        setattr(self.model, x_param, original_x_value)
        self.model._update_d_values()
        
        # Set layout
        if title is None:
            title = f"{GREEK_LABELS[greek]} vs {PARAM_LABELS[x_param]}"
            if shift_param != x_param and len(shifts) > 1:
                title += f" (Multiple {PARAM_LABELS[shift_param]} Shifts)"
        
        # Add option information footer
        option_type_text = "Call" if self.model.option_type == "call" else "Put"
        footer_text = (f"{option_type_text} Option - S: {self.model.S}, K: {self.model.K}, "
                      f"T: {self.model.T:.2f}, r: {self.model.r:.2%}, σ: {self.model.sigma:.2%}")
        
        fig.update_layout(
            title=title,
            xaxis_title=PARAM_LABELS[x_param],
            yaxis_title=GREEK_LABELS[greek],
            template="plotly_white",
            annotations=[
                dict(
                    text=footer_text,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=10)
                )
            ]
        )
        
        return fig
    
    def plot_greek_curves(
        self, 
        greek_name: str, 
        greek_func: Callable, 
        param_name: str, 
        param_range: List[float],
        shift_param_name: str,
        shift_values: List[float],
        title: str = None
    ) -> go.Figure:
        """
        Plot multiple Greek curves with various parameter shifts.
        
        Args:
            greek_name: Name of the Greek for labeling
            greek_func: Function that calculates the Greek
            param_name: Primary parameter to vary (x-axis)
            param_range: Range of values for primary parameter
            shift_param_name: Secondary parameter to shift for multiple curves
            shift_values: List of values for the shift parameter
            title: Optional plot title
            
        Returns:
            Plotly figure with multiple curves
        """
        fig = self.go.Figure()
        
        original_shift_value = getattr(self.model, shift_param_name)
        
        for shift_value in shift_values:
            # Set the shift parameter
            setattr(self.model, shift_param_name, shift_value)
            self.model._update_d_values()
            
            # Compute the curve
            data = self.compute_greek_curve(greek_func, param_name, param_range)
            
            # Plot the curve
            label = f"{shift_param_name} = {shift_value:.2f}"
            fig.add_trace(
                self.go.Scatter(
                    x=data['param_values'],
                    y=data['greek_values'],
                    mode='lines',
                    name=label
                )
            )
        
        # Restore the original shift parameter value
        setattr(self.model, shift_param_name, original_shift_value)
        self.model._update_d_values()
        
        # Set layout
        fig.update_layout(
            title=title or f'{greek_name} vs {PARAM_LABELS.get(param_name, param_name)} for different {shift_param_name} values',
            xaxis_title=PARAM_LABELS.get(param_name, param_name),
            yaxis_title=greek_name,
            template="plotly_white"
        )
        
        return fig
    
    def generate_summary_table(self) -> Dict[str, float]:
        """Generate a summary of all Greeks for the current parameter values."""
        return {
            'Price': self.model.price(),
            'Delta': self.model.delta(),
            'Gamma': self.model.gamma(),
            'Vega': self.model.vega(),
            'Theta': self.model.theta(),
            'Rho': self.model.rho(),
            'Vanna': self.model.vanna(),
        }
    
    def plot_multiple_greeks(
        self, 
        greeks: List[str], 
        x_param: str, 
        shifts: List[float] = None,
        shift_param: str = None,
        range_pct: float = 0.2, 
        points: int = 100
    ) -> go.Figure:
        """
        Plot multiple Greeks against the same parameter.
        
        Args:
            greeks (List[str]): List of Greeks to plot ('delta', 'gamma', etc.).
            x_param (str): The parameter for the x-axis ('S', 'K', 'T', 'r', 'sigma').
            shifts (List[float], optional): List of shifts to apply.
            shift_param (str, optional): Parameter to shift.
            range_pct (float): The range to vary the parameter by, as a percentage.
            points (int): Number of points in the curve.
            
        Returns:
            Plotly figure with the plotted subplots
        """
        if len(greeks) == 0:
            raise ValueError("Must specify at least one Greek to plot")
            
        # Create subplot grid based on number of Greeks
        n_greeks = len(greeks)
        n_cols = min(3, n_greeks)
        n_rows = (n_greeks + n_cols - 1) // n_cols
        
        subplot_titles = [GREEK_LABELS.get(greek, greek) for greek in greeks]
        fig = self.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
        
        # Store original values to restore later
        original_values = {}
        for param in PARAM_LABELS.keys():
            if hasattr(self.model, param):
                original_values[param] = getattr(self.model, param)
        
        # Default shift values if none provided
        if shifts is None:
            shifts = [0.0]
            
        if shift_param is None:
            shift_param = x_param
        
        # Plot each Greek
        for i, greek in enumerate(greeks):
            # Calculate row and column for current subplot
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Get the corresponding Greek function
            greek_func = getattr(self.model, greek.lower())
            
            # For each shift, compute and plot the curve
            for shift in shifts:
                # Apply shift to the shift parameter if different from x_param
                if shift_param != x_param:
                    shift_value = original_values[shift_param] * (1 + shift)
                    setattr(self.model, shift_param, shift_value)
                    self.model._update_d_values()
                
                # Create x-axis parameter range
                if shift_param == x_param:
                    # If shifting x-axis parameter, adjust the range
                    x_range = np.linspace(
                        original_values[x_param] * (1 - range_pct) * (1 + shift),
                        original_values[x_param] * (1 + range_pct) * (1 + shift),
                        points
                    )
                else:
                    # Normal range centered around original value
                    x_range = np.linspace(
                        original_values[x_param] * (1 - range_pct),
                        original_values[x_param] * (1 + range_pct),
                        points
                    )
                
                # Compute the curve
                data = self.compute_greek_curve(greek_func, x_param, x_range)
                
                # Generate label
                if shift == 0:
                    label = f"Base {PARAM_LABELS[shift_param]}"
                else:
                    shift_pct = shift * 100
                    label = f"{PARAM_LABELS[shift_param]} {'+' if shift > 0 else ''}{shift_pct:.0f}%"
                
                # Only show legend in the first plot
                showlegend = (i == 0)
                
                # Plot the curve
                fig.add_trace(
                    self.go.Scatter(
                        x=data['param_values'], 
                        y=data['greek_values'], 
                        name=label,
                        showlegend=showlegend
                    ),
                    row=row, col=col
                )
        
        # Restore original parameter values
        for param, value in original_values.items():
            setattr(self.model, param, value)
        self.model._update_d_values()
        
        # Add overall title and adjust layout
        shift_info = ""
        if shift_param != x_param and len(shifts) > 1:
            shift_info = f" (Multiple {PARAM_LABELS[shift_param]} Shifts)"
            
        # Add option information footer
        option_type_text = "Call" if self.model.option_type == "call" else "Put"
        footer_text = (f"{option_type_text} Option - S: {self.model.S}, K: {self.model.K}, "
                      f"T: {self.model.T:.2f}, r: {self.model.r:.2%}, σ: {self.model.sigma:.2%}")
        
        fig.update_layout(
            title=f"Option Greeks vs {PARAM_LABELS[x_param]}{shift_info}",
            template="plotly_white",
            height=250 * n_rows,
            annotations=[
                dict(
                    text=footer_text,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=10)
                )
            ]
        )
        
        # Update x and y axis labels for all subplots
        for i in range(1, n_rows * n_cols + 1):
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1
            
            if row == n_rows:  # Only add x-axis titles to bottom row
                fig.update_xaxes(title_text=PARAM_LABELS[x_param], row=row, col=col)
            
            # Add y-axis titles to all plots
            if col == 1:  # Only add y-axis titles to leftmost column
                greek_idx = (row - 1) * n_cols + col - 1
                if greek_idx < len(greeks):
                    fig.update_yaxes(title_text=GREEK_LABELS[greeks[greek_idx]], row=row, col=col)
        
        return fig
    
    def make_subplots(self, rows, cols, subplot_titles=None):
        """Helper method to create subplot grid."""
        from plotly.subplots import make_subplots
        return make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    
    def plot_summary_dashboard(self) -> go.Figure:
        """Create a dashboard with multiple plots showing key Greeks."""
        # Create 2x3 subplots
        fig = self.make_subplots(
            rows=2, cols=3,
            subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna')
        )
        
        # Spot price range centered around current spot price
        spot_range = np.linspace(self.model.S * 0.7, self.model.S * 1.3, 100)
        
        # Delta vs Spot
        self.add_greek_to_subplot(fig, 'Delta', self.model.delta, 'S', spot_range, 1, 1)
        
        # Gamma vs Spot
        self.add_greek_to_subplot(fig, 'Gamma', self.model.gamma, 'S', spot_range, 1, 2)
        
        # Vega vs Spot
        self.add_greek_to_subplot(fig, 'Vega', self.model.vega, 'S', spot_range, 1, 3)
        
        # Theta vs Spot
        self.add_greek_to_subplot(fig, 'Theta', self.model.theta, 'S', spot_range, 2, 1)
        
        # Rho vs Spot
        self.add_greek_to_subplot(fig, 'Rho', self.model.rho, 'S', spot_range, 2, 2)
        
        # Vanna vs Spot
        self.add_greek_to_subplot(fig, 'Vanna', self.model.vanna, 'S', spot_range, 2, 3)
        
        # Set layout
        option_type_text = "Call" if self.model.option_type == "call" else "Put"
        title = f'Option Greeks for {option_type_text} Option (K={self.model.K}, T={self.model.T:.2f}, r={self.model.r:.2%}, σ={self.model.sigma:.2%})'
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=700,
            showlegend=False
        )
        
        # Add x-axis titles to bottom row only
        for col in range(1, 4):
            fig.update_xaxes(title_text='Spot Price (S)', row=2, col=col)
        
        return fig
    
    def add_greek_to_subplot(self, fig, greek_name, greek_func, param_name, param_range, row, col):
        """Helper to compute and plot a Greek on a given subplot."""
        data = self.compute_greek_curve(greek_func, param_name, param_range)
        
        fig.add_trace(
            self.go.Scatter(
                x=data['param_values'],
                y=data['greek_values'],
                line=dict(color='blue', width=2),
                name=greek_name
            ),
            row=row, col=col
        )