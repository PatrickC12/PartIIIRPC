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

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

class CSVPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Plotter")

        # Frame for file selection and options
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Button to load folder
        self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=5)

        # Dictionary to hold the checkboxes and file paths
        self.csv_files = {}
        
        # Plot button, disabled initially
        self.plot_button = ttk.Button(self.frame, text="Plot Selected", state='disabled', command=self.plot_selected)
        self.plot_button.pack(pady=5)
    
    def find_best_threshold(self, data):
        # Placeholder values for best error and threshold
        best_error = float('inf')
        best_threshold = None

        # Try thresholds in the voltage range
        for threshold in np.linspace(data['Voltage/kV'].min(), data['Voltage/kV'].max(), 100):
            # Split data into linear and exponential parts
            linear_data = data[data['Voltage/kV'] <= threshold]
            exp_data = data[data['Voltage/kV'] > threshold]

            if len(linear_data) < 2 or len(exp_data) < 3:  # Ensure exp_data has at least 3 points for 3 parameters
                # Not enough points to fit
                continue

            # Fit linear part
            linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
            linear_fit_func = np.poly1d(linear_fit_params)

            # Fit exponential part, now with check for sufficient data points
            try:
                exp_fit_params, _ = curve_fit(exp_func, exp_data['Voltage/kV'], exp_data['Current/uA'])
            except RuntimeError:
                # Fit failed
                continue

            # Calculate combined error (sum of squared residuals)
            linear_resid = linear_data['Current/uA'] - linear_fit_func(linear_data['Voltage/kV'])
            exp_resid = exp_data['Current/uA'] - exp_func(exp_data['Voltage/kV'], *exp_fit_params)
            total_error = np.sum(linear_resid**2) + np.sum(exp_resid**2)

            if total_error < best_error:
                best_error = total_error
                best_threshold = threshold

        return best_threshold


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

        # Create and pack the error bar checkbox after loading the files
        self.error_bar_var = tk.BooleanVar()
        self.error_bar_chk = ttk.Checkbutton(self.frame, text="Add Error Bars", variable=self.error_bar_var)
        self.error_bar_chk.pack(pady=5)
        
        # checkbox for line fitting
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
                # Fit linear region
                threshold = self.find_best_threshold(data[['Voltage/kV', 'Current/uA']])
                if threshold is not None:
                    
                    linear_data = data[data['Voltage/kV'] <= threshold]  # Define a suitable threshold
                    linear_fit_params = np.polyfit(linear_data.iloc[:, 0], linear_data.iloc[:, 1], 1)
                    linear_fit_line = np.poly1d(linear_fit_params)
                    # LowerValues = [x for x in data.iloc[:, 0] if x < threshold]
                    # plt.plot(data.iloc[:, 0], linear_fit_line(data.iloc[:, 0]), label=f'Linear Fit: y={linear_fit_params[0]:.2f}x+{linear_fit_params[1]:.2f}')
                    plt.plot([x for x in data.iloc[:, 0] if x < threshold], linear_fit_line([x for x in data.iloc[:, 0] if x < threshold]), label=f'Linear Fit: y={linear_fit_params[0]:.2f}x+{linear_fit_params[1]:.2f}')

                    # Fit exponential region
                    # Assuming exponential region is defined by higher voltages
                    def exp_func(x, a, b, c):
                        return a * np.exp(b * x) + c
                    exp_data = data[data['Voltage/kV'] > threshold]  # Same threshold or adjust as necessary
                    exp_fit_params, _ = curve_fit(exp_func, exp_data.iloc[:, 0], exp_data.iloc[:, 1])
                    exp_data_x_values = exp_data['Voltage/kV'].values  # Convert to numpy array for arithmetic operations
                    exp_fit_y_values = exp_func(exp_data_x_values, *exp_fit_params)  # Apply the function to the numpy array directly
                    plt.plot(exp_data_x_values, exp_fit_y_values, label=f'Exp Fit: y={exp_fit_params[0]:.2f}e^({exp_fit_params[1]:.2f}x)+{exp_fit_params[2]:.2f}')
                    # plt.plot(data.iloc[:, 0], exp_func(data.iloc[:, 0], *exp_fit_params), label=f'Exp Fit: y={exp_fit_params[0]:.2f}e^({exp_fit_params[1]:.2f}x)+{exp_fit_params[2]:.2f}')
                    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], label=file)
                else:
                    messagebox.showwarning("Fit Error", "Could not determine a fitting threshold.")
                    return
            
            elif self.error_bar_var.get():
                # Adjusting error values: replace 0 with 0.01
                errors = data.iloc[:, 2].replace(0, 0.01)
                sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], label=file)
                plt.errorbar(data.iloc[:, 0], data.iloc[:, 1], yerr=errors, fmt='o', capsize=5, label='_nolegend_')
            else:
                sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], label=file)

        plt.xlabel('Voltage (kV)')
        plt.ylabel('Current (uA)')
        plt.title('Voltage vs. Current')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()
