
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

def exponential_model(t, x_infinity, tau):
    """
    Exponential model function: x(t) = x_infinity * (1 - exp(-t/tau))
    
    Args:
        t: Time variable
        x_infinity: Asymptotic displacement value
        tau: Time constant
        
    Returns:
        Displacement value at time t
    """
    return x_infinity * (1 - np.exp(-t/tau))
# Input parameters
# Path to your CSV data file - update this to your actual path
input_folder = os.path.dirname("...")
csv_path = os.path.join(input_folder, "filename.csv")
# Check if file exists
if not os.path.exists(csv_path):
    print(f"Data file not found: {csv_path}")
    print("Please provide the correct path to your displacement_data.csv file.")
    exit()
# Read and prepare the data
try:
    # Read CSV data
    data = pd.read_csv(csv_path)
    print("Data loaded successfully")
    
    # Extract time and displacement columns
    time_points = data['Time (min)'].values
    displacements = data['Displacement (mm)'].values
    
    # Make sure displacements are positive
    displacements = np.abs(displacements)
    
    print(f"Found {len(time_points)} data points")
except Exception as e:
    print(f"Error reading data: {e}")
    exit()
# Fit the exponential model to the data
try:
    # Initial parameter guesses
    # x_infinity: estimate as the maximum displacement
    # tau: estimate as 1/3 of the total time range
    initial_x_infinity = np.max(displacements)
    initial_tau = np.max(time_points) / 3
    
    # Set bounds to ensure physical meaning (positive values)
    bounds = ([0, 0], [np.inf, np.inf])
    
    print(f"Fitting exponential model: x(t) = x_infinity * (1 - exp(-t/tau))")
    print(f"   Initial guess: x_infinity ≈ {initial_x_infinity:.4f} mm, tau ≈ {initial_tau:.4f} min")
    
    # Perform the curve fitting
    popt, pcov = curve_fit(
        exponential_model, 
        time_points, 
        displacements, 
        p0=[initial_x_infinity, initial_tau],
        bounds=bounds,
        maxfev=10000  # Increase max function evaluations for complex fits
    )
    
    # Extract fitted parameters
    x_infinity_fit, tau_fit = popt
    
    # Calculate parameter uncertainties from covariance matrix
    perr = np.sqrt(np.diag(pcov))
    x_infinity_err, tau_err = perr
    
    print(f"Fitting successful!")
    print(f"Fitted parameters:")
    print(f"   x_infinity = {x_infinity_fit:.6f} ± {x_infinity_err:.6f} mm")
    print(f"   tau = {tau_fit:.6f} ± {tau_err:.6f} min")
    print(f"   (The ± values represent standard deviation uncertainty)")
    
    # Calculate fitted curve for plotting
    t_smooth = np.linspace(0, np.max(time_points), 1000)
    y_fit = exponential_model(t_smooth, x_infinity_fit, tau_fit)
    
    # Calculate coefficient of determination (R²)
    y_fit_at_data = exponential_model(time_points, x_infinity_fit, tau_fit)
    ss_tot = np.sum((displacements - np.mean(displacements))**2)
    ss_res = np.sum((displacements - y_fit_at_data)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Goodness of fit: R² = {r_squared:.6f}")
    
except Exception as e:
    print(f"Error during fitting: {e}")
    print("Try adjusting initial parameter guesses or using a different model if the fit fails.")
    exit()
# === Create the plot ===
plt.figure(figsize=(12, 8))
# Plot original data
plt.scatter(time_points, displacements, color='blue', label='Measured Data', alpha=0.7)
# Plot fitted curve
plt.plot(t_smooth, y_fit, 'r-', linewidth=2, label=f'Fitted: x(t) = {x_infinity_fit:.4f} * (1 - exp(-t/{tau_fit:.4f}))')
# Add grid and labels
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Displacement (mm)', fontsize=12)
plt.title('Exponential Fit to Displacement Data (4% Agarose Gel Sample under 130 mbar Pressure)', fontsize=14)
# Add fitted parameter text box
textstr = '\n'.join((
    f'Fitted Parameters:',
    f'x_∞ = {x_infinity_fit:.4f} ± {x_infinity_err:.4f} mm',
    f'τ = {tau_fit:.4f} ± {tau_err:.4f} min',
    f'R² = {r_squared:.4f}'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
plt.legend(loc='lower right')
plt.tight_layout()

# Show the plot (comment this out if running in a non-interactive environment)
plt.show()
print("Analysis completed successfully!")
