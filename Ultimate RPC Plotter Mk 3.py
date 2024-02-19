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

#Define Fitting function

def exp_func(x, a, b, c):
    return a * np.exp(b * (x-c))

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
        best_R2_linear = -float('inf')
        best_R2_exp = -float('inf')
        best_threshold_linear = None
        best_threshold_exp = None

        # Define a range of potential thresholds
        thresholds = np.linspace(data['Voltage/kV'].min(), data['Voltage/kV'].max(), 100)

        # Iterate over thresholds for the linear region
        for i, threshold_linear in enumerate(thresholds[:-1]):  # Exclude the last point to ensure a separate region for exponential fit
            linear_data = data[data['Voltage/kV'] <= threshold_linear]

            if len(linear_data) < 2:
                continue

            # Linear fit and R^2 calculation for the linear region
            linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
            linear_fit_func = np.poly1d(linear_fit_params)
            SSE_linear = np.sum((linear_data['Current/uA'] - linear_fit_func(linear_data['Voltage/kV']))**2)
            SST_linear = np.sum((linear_data['Current/uA'] - linear_data['Current/uA'].mean())**2)
            R2_linear = 1 - SSE_linear/SST_linear

            if R2_linear > best_R2_linear:
                best_R2_linear = R2_linear
                best_threshold_linear = threshold_linear

            # Iterate over thresholds for the exponential region, starting from the current threshold_linear + 1 to ensure separation
            for threshold_exp in thresholds[i+1:]:
                exp_data = data[data['Voltage/kV'] > threshold_exp]

                if len(exp_data) < 2:
                    continue

                # Exponential fit and R^2 calculation
                try:
                    exp_fit_params, _ = curve_fit(lambda x, a, b: exp_func(x, a, b, threshold_exp), exp_data['Voltage/kV'], exp_data['Current/uA'])
                    exp_resid = np.log(exp_data['Current/uA']) - np.log(exp_func(exp_data['Voltage/kV'], *exp_fit_params, threshold_exp))
                    SSE_exp = np.sum(exp_resid**2)
                    SST_exp = np.sum((np.log(exp_data['Current/uA']) - np.log(exp_data['Current/uA']).mean())**2)
                    R2_exp = 1 - SSE_exp/SST_exp
                except RuntimeError:
                    continue

                if R2_exp > best_R2_exp and threshold_exp > best_threshold_linear:
                    best_R2_exp = R2_exp
                    best_threshold_exp = threshold_exp

        return best_threshold_linear, best_threshold_exp



    def load_folder(self):
        global folder_path
        
        #Folder containing the csv data

        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        if not folder_path:
            return
        
        for widget in self.frame.winfo_children():
            if widget not in [self.load_button, self.plot_button]:
                widget.destroy()

        self.csv_files = {}

        files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        #Check if any csv files in directory chosen

        if not files:
            messagebox.showinfo("No CSV Files", "No CSV files found in the selected folder.")
            return
        
        #Enable plot_button widget after data folder selected

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
                threshold = self.find_best_threshold(data[['Voltage/kV', 'Current/uA']])
                if threshold is not None:
                    # Fit linear part
                    linear_data = data[data['Voltage/kV'] <= threshold]
                    linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
                    linear_fit_func = np.poly1d(linear_fit_params)
                    
                    # Plot linear fit
                    linear_x_vals = np.linspace(data['Voltage/kV'].min(), threshold, 100)
                    plt.plot(linear_x_vals, linear_fit_func(linear_x_vals), label=f'Linear Fit: y={linear_fit_params[0]:.2f}x+{linear_fit_params[1]:.2f}', linestyle="--")

                    # Fit exponential part
                    exp_data = data[data['Voltage/kV'] > threshold]
                    try:
                        c_fixed = threshold
                        exp_fit_params, _ = curve_fit(lambda x, a, b: exp_func(x, a, b, c_fixed), exp_data['Voltage/kV'], exp_data['Current/uA'])
                    except RuntimeError:
                        continue  # Skip if fit fails

                    # Plot exponential fit
                    exp_x_vals = np.linspace(threshold, data['Voltage/kV'].max(), 100)
                    plt.plot(exp_x_vals, exp_func(exp_x_vals, *exp_fit_params, c_fixed), label='Exponential Fit', linestyle="--")

                    # Plot original data points
                    sns.scatterplot(x=data['Voltage/kV'], y=data['Current/uA'], label=file)
                    
                else:
                    messagebox.showwarning("Fit Error", "Could not determine a fitting threshold.")
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
