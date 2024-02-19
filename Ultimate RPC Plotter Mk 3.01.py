import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.optimize import curve_fit

# Setting the Seaborn theme
sns.set_theme(style="darkgrid")

# Define Fitting function
def exp_func(x, a, b, c):
    return a * np.exp(b * (x - c))

class CSVPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Plotter")
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=5)

        self.csv_files = {}
        self.plot_button = ttk.Button(self.frame, text="Plot Selected", state='disabled', command=self.plot_selected)
        self.plot_button.pack(pady=5)

    def find_best_threshold(self, data):
        best_R2_linear = -np.inf
        best_R2_exp = -np.inf
        best_threshold_linear = None
        best_threshold_exp = None

        # Precompute common values and use efficient numpy operations
        voltage = data['Voltage/kV'].values
        current = data['Current/uA'].values
        log_current = np.log(current)

        thresholds = np.linspace(voltage.min(), voltage.max(), 100)

        for threshold_linear in thresholds[:-1]:  # Exclude the last point
            linear_mask = voltage <= threshold_linear
            linear_voltages = voltage[linear_mask]
            linear_currents = current[linear_mask]
            # log_linear_currents = log_current[linear_mask]

            if len(linear_voltages) < 2:
                continue

            # Linear fit (using numpy for efficiency)
            A = np.vstack([linear_voltages, np.ones(len(linear_voltages))]).T
            linear_fit_params = np.linalg.lstsq(A, linear_currents, rcond=None)[0]
            linear_fit_residuals = linear_currents - (linear_fit_params[0] * linear_voltages + linear_fit_params[1])
            R2_linear = 1 - np.var(linear_fit_residuals) / np.var(linear_currents)

            if R2_linear > best_R2_linear and R2_linear < 0.975:
                best_R2_linear = R2_linear
                best_threshold_linear = threshold_linear

            exp_mask = voltage > threshold_linear
            exp_voltages = voltage[exp_mask]
            exp_currents = current[exp_mask]
            log_exp_currents = log_current[exp_mask]

            if len(exp_voltages) < 2:
                continue

            try:
                exp_fit_params, _ = curve_fit(lambda x, a, b: a * np.exp(b * x), exp_voltages, exp_currents, maxfev=10000)
                exp_fit_residuals = log_exp_currents - np.log(exp_fit_params[0] * np.exp(exp_fit_params[1] * exp_voltages))
                R2_exp = 1 - np.var(exp_fit_residuals) / np.var(log_exp_currents)
            except RuntimeError:
                continue

            if R2_exp > best_R2_exp and R2_exp < 1:
                best_R2_exp = R2_exp
                best_threshold_exp = threshold_linear

        return best_threshold_linear, best_threshold_exp


    def load_folder(self):
        global folder_path
        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        if not folder_path:
            return

        for widget in self.frame.winfo_children():
            if widget not in [self.load_button, self.plot_button]:
                widget.destroy()

        self.csv_files = {}
        files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        if not files:
            messagebox.showinfo("No CSV Files", "No CSV files found in the selected folder.")
            return

        self.plot_button['state'] = 'normal'
        for file in files:
            self.csv_files[file] = tk.BooleanVar()
            chk = ttk.Checkbutton(self.frame, text=file, variable=self.csv_files[file])
            chk.pack(anchor='w')

        # Additional UI elements for error bars and fitting lines
        self.error_bar_var = tk.BooleanVar()
        self.error_bar_chk = ttk.Checkbutton(self.frame, text="Add Error Bars", variable=self.error_bar_var)
        self.error_bar_chk.pack(pady=5)

        self.fit_lines_var = tk.BooleanVar()
        self.fit_lines_chk = ttk.Checkbutton(self.frame, text="Fit Linear and Exponential Regions", variable=self.fit_lines_var)
        self.fit_lines_chk.pack(pady=5)

    def plot_selected(self):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return
        
        plt.figure(figsize=(10, 8))

        for file in selected_files:
            data_path = os.path.join(folder_path, file)
            data = pd.read_csv(data_path)

            if self.fit_lines_var.get():
                threshold_linear, threshold_exp = self.find_best_threshold(data[['Voltage/kV', 'Current/uA']])
                if threshold_linear is not None and threshold_exp is not None:
                    # Fit linear part
                    linear_data = data[data['Voltage/kV'] <= threshold_linear]
                    linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
                    linear_fit_func = np.poly1d(linear_fit_params)
                    
                    # Plot linear fit
                    linear_x_vals = np.linspace(data['Voltage/kV'].min(), threshold_linear, 100)
                    plt.plot(linear_x_vals, linear_fit_func(linear_x_vals), label=f'Linear Fit: y={linear_fit_params[0]:.2f}x+{linear_fit_params[1]:.2f}', linestyle="--")

                    # Fit exponential part
                    exp_data = data[data['Voltage/kV'] > threshold_exp]  # Use threshold_exp for the exponential region
                    try:
                        exp_fit_params, _ = curve_fit(exp_func, exp_data['Voltage/kV'], exp_data['Current/uA'], p0=[1.0, 1.0, threshold_exp])
                    except RuntimeError:
                        continue  # Skip if fit fails

                    # Plot exponential fit
                    exp_x_vals = np.linspace(threshold_exp - 1, data['Voltage/kV'].max(), 100)
                    plt.plot(exp_x_vals, exp_func(exp_x_vals, *exp_fit_params), label=f'exponential Fit: y={exp_fit_params[0]:.2f}exp({exp_fit_params[1]:.2f}(x-{threshold_exp:.2f}))', linestyle="--")


                    # Plot original data points
                    sns.scatterplot(x=data['Voltage/kV'], y=data['Current/uA'], label=file)
                    
                else:
                    messagebox.showwarning("Fit Error", "Could not determine fitting thresholds.")
                    return
            elif self.error_bar_var.get():
                # Your error bar plotting logic here
                pass
            else:
                # Plot without fitting or error bars
                sns.scatterplot(x=data['Voltage/kV'], y=data['Current/uA'], label=file)

        plt.xlabel('Voltage (kV)')
        plt.ylabel('Current (uA)')
        plt.title('Voltage vs. Current')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()
