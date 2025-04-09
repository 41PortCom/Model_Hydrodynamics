import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import scipy.signal as spsig

# -------------------------------------------------------------------
# Physical and geometric parameters
# -------------------------------------------------------------------
R = 0.025        # Cylinder radius (m)
U = 0.13         # Flow velocity (m/s)
rho = 1000.0     # Density (e.g., water)
p_infty = 0.0    # Reference pressure

# For cylinder displacement:
# The center in the x-direction is fixed at x_c, and the y-position varies from -0.1 to 1.5 m over 301 frames
x_c = 0.05
y_values = np.linspace(-0.18, 2.0, 301)  # Positions in m (for each frame)

# Compute time (in seconds) using the velocity U
# The time corresponds to the duration required to travel the distance from y_values[0]
t_values = (y_values - y_values[0]) / U

# -------------------------------------------------------------------
# Pressure calculation functions (based on potential flow)
# -------------------------------------------------------------------
def pressure_potential_flow(x, y, x_c, y_c, R, U, rho, p_ref=0.0):
    # Calculate the pressure at (x,y) for a cylinder located at (x_c, y_c)
    X = x - x_c
    Y = y - y_c
    r = np.sqrt(X**2 + Y**2)
    if r < 1e-12:
        return np.nan
    theta = np.arctan2(X, Y)
    v_r = U * np.cos(theta) * (1.0 - (R**2) / (r**2))
    v_theta = -U * np.sin(theta) * (1.0 + (R**2) / (r**2))
    v2 = v_r**2 + v_theta**2
    p = p_ref + 0.5 * rho * (U**2 - v2)
    return p

def pressure_field(xgrid, ygrid, x_c, y_c, R, U, rho, p_ref=0.0):
    # Compute the pressure field on the grid
    X = xgrid - x_c
    Y = ygrid - y_c
    r = np.sqrt(X**2 + Y**2)
    r = np.where(r < 1e-12, 1e-12, r)
    theta = np.arctan2(X, Y)
    v_r = U * np.cos(theta) * (1.0 - (R**2) / (r**2))
    v_theta = -U * np.sin(theta) * (1.0 + (R**2) / (r**2))
    v2 = v_r**2 + v_theta**2
    p = p_ref + 0.5 * rho * (U**2 - v2)
    p[r < R] = np.nan
    return p

# -------------------------------------------------------------------
# Pre-calculation of the pressure curve (at (0,0)) for each frame
# -------------------------------------------------------------------
p_classique_array = []
for y_c in y_values:
    p_classique_array.append(pressure_potential_flow(0.0, 0.0, x_c, y_c, R, U, rho, p_infty))
p_classique_array = np.array(p_classique_array)

# -------------------------------------------------------------------
# Create a grid for the pressure field (animation zone)
# -------------------------------------------------------------------
x_min, x_max = -0.2, 0.2
y_min, y_max = -0.2, 0.2
nx, ny = 500, 500  # Grid resolution
x_lin = np.linspace(x_min, x_max, nx)
y_lin = np.linspace(y_min, y_max, ny)
Xgrid, Ygrid = np.meshgrid(x_lin, y_lin)

def lowpass_filter(data, fs, cutoff, order=4):
    """
    Applies a Butterworth low-pass filter.

    :param data: Signal to be filtered (numpy array)
    :param fs: Sampling frequency (Hz)
    :param cutoff: Cutoff frequency (Hz)
    :param order: Filter order (default=4)
    :return: Filtered signal (numpy array)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = spsig.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = spsig.filtfilt(b, a, data)
    return filtered_data

def plot_selected_experiments(selected_indices, filtered_starts, df, min_interval):
    # Common time grid for interpolation (500 points between 0 and 20 s)
    common_time = np.linspace(0, 20, num=500)
    signals = ["R", "Sensor0"]

    # Dictionary to store the interpolated signals to compute the average
    average_signals = {}
    
    # For each signal ("R" and "Sensor0"), plot the curves of the selected experiments
    for signal in signals:
        plt.figure(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, num=len(selected_indices)))
        interpolated_signals = []  # List to accumulate interpolated signals
        
        for j, idx in enumerate(selected_indices):
            start = filtered_starts[idx]
            # Extract the data for the selected experiment
            exp_df = df[(df["timestamp"] >= start) & (df["timestamp"] < start + min_interval)].copy()
            exp_df["relative_time"] = exp_df["timestamp"] - start
            
            if signal == "R":
                # Interpolate the "R" and "Sensor0" signals on the common grid
                R_interp = np.interp(common_time, exp_df["relative_time"], exp_df["R"])
                sensor0_interp = np.interp(common_time, exp_df["relative_time"], exp_df["Sensor0"])
                # Compute the optimal coefficient k using the least squares method
                k = np.sum(R_interp * sensor0_interp) / np.sum(R_interp ** 2)
                # Adjust the amplitude of R
                interp_signal = R_interp * k
            else:
                interp_signal = np.interp(common_time, exp_df["relative_time"], exp_df[signal])
            
            interpolated_signals.append(interp_signal)
            plt.plot(common_time, interp_signal, color=colors[j], alpha=0.7, label=f"Exp {idx+1}")
        
        plt.xlabel("Time (s)")
        plt.ylabel(f"Sensor value (Pa)")
        plt.title(f"Superimposed signals of the n experiments")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Compute the average of the signal for the selected experiments
        if interpolated_signals:
            average_signals[signal] = np.mean(interpolated_signals, axis=0)
    
    # Display a plot containing the average of the two signals
    plt.figure(figsize=(10, 5))
    if "R" in average_signals:
        plt.plot(common_time, average_signals["R"],
                 label="Data from the wave sensor", linewidth=2)
    if "Sensor0" in average_signals:
        plt.plot(common_time, average_signals["Sensor0"],
                 label="Data from the pressure sensor", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title("The experiments")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Compute and display the (Sensor0 - R) signal averaged and filtered by a low-pass (for instance 3 Hz)
    if "R" in average_signals and "Sensor0" in average_signals:
        diff_signal = average_signals["Sensor0"] - average_signals["R"]
        # Sampling frequency: 500 points over 20 s => fs = 25 Hz
        fs = len(common_time) / (common_time[-1] - common_time[0])
        # We can adjust the cutoff frequency and filter order here if needed
        filtered_diff = lowpass_filter(diff_signal, fs, cutoff=3, order=4)
        
        plt.figure(figsize=(10, 5))
        plt.plot(common_time, filtered_diff,
                 label="(Pressure sensor - Wave sensor) filtered (3Hz)", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Filtered difference (Pa)")
        plt.title("Signal (Pressure sensor - Wave sensor) with a 3 Hz low-pass filter")
        plt.legend()
        plt.grid()
        plt.show()

        # Compute the mathematical model interpolated on the common time grid
        model_interp = np.interp(common_time, t_values, p_classique_array)
        
        # Plot the comparison between the filtered measured signal and the mathematical model
        plt.figure(figsize=(10, 5))
        plt.plot(common_time, filtered_diff,
                 label="(Pressure sensor - Wave sensor) filtered (3Hz)", linewidth=2)
        plt.plot(common_time, model_interp,
                 label="Mathematical model", linewidth=2, linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Pa)")
        plt.title("Comparison: (Pressure sensor - Wave sensor) filtered (3Hz) vs. Mathematical model")
        plt.legend()
        plt.grid()
        plt.show()

    # Save the average data in a CSV file in the same format as the input
    save_path = filedialog.asksaveasfilename(
        title="Save the average data",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if save_path:
        df_avg = pd.DataFrame({
            "timestamp": common_time,          # In seconds (multiply by 1000 for ms if needed)
            "R": average_signals.get("R", np.nan),
            "Sensor0": average_signals.get("Sensor0", np.nan)
        })
        df_avg.to_csv(save_path, index=False, sep=';')
        print(f"Average data saved in {save_path}")

def main():
    # Create the root window and hide it for file selection
    root = tk.Tk()
    root.withdraw()
    
    # Dialog box to select a CSV file
    file_path = filedialog.askopenfilename(
        title="Select a CSV file", filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("No file selected.")
        return
    
    try:
        # Load the data
        df = pd.read_csv(file_path, sep=';')
        # Check the presence of the expected columns
        required_columns = ["timestamp", "R"] + [f"Sensor{i}" for i in range(8)]
        if not all(col in df.columns for col in required_columns):
            print("The CSV file does not contain the required columns.")
            return
        
        # Convert timestamps to seconds (if originally in milliseconds)
        df["timestamp"] = df["timestamp"] / 1000
        
        # Detect experiments based on Sensor0 (threshold choice)
        threshold = 1.5       # Threshold to detect an experiment
        min_interval = 25     # Minimum interval between two experiments (in seconds)
        df["exp_detected"] = df["Sensor0"] > threshold
        df["exp_start"] = df["exp_detected"] & ~df["exp_detected"].shift(1, fill_value=False)
        experiment_starts = df.loc[df["exp_start"], "timestamp"].tolist()
        
        # Filter the experiment start times to respect the minimum interval
        filtered_starts = []
        last_start = -min_interval
        for start in experiment_starts:
            if start - last_start >= min_interval:
                filtered_starts.append(start)
                last_start = start
        
        num_experiments = len(filtered_starts)
        print(f"Number of detected experiments: {num_experiments}")
        
        # Create a selection window for the experiments (Toplevel)
        selection_window = tk.Toplevel(root)
        selection_window.title("Select the experiments to display")
        
        # Create a checkbox for each detected experiment
        vars_experiments = []
        for i in range(num_experiments):
            var = tk.IntVar(value=1)  # Box checked by default
            chk = tk.Checkbutton(selection_window, text=f"Experiment {i+1}", variable=var)
            chk.pack(anchor="w", padx=10, pady=2)
            vars_experiments.append(var)
        
        # Function called when validating the selection
        def on_validate():
            selected_indices = [i for i, var in enumerate(vars_experiments) if var.get() == 1]
            if not selected_indices:
                print("No experiment selected.")
                return
            selection_window.destroy()  # Close the selection window
            plot_selected_experiments(selected_indices, filtered_starts, df, min_interval)
        
        validate_btn = tk.Button(selection_window, text="Validate", command=on_validate)
        validate_btn.pack(pady=10)
        
        selection_window.mainloop()
        
    except Exception as e:
        print(f"Error while reading the CSV file: {e}")

if __name__ == "__main__":
    main()
