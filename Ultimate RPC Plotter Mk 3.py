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
        # Placeholder values for best error and threshold
        best_error = float('inf')
        best_threshold = None

        # Try thresholds in the voltage range
        for threshold in np.linspace(data['Voltage/kV'].min(), data['Voltage/kV'].max(), 100):
            # Split data into linear and exponential parts
            linear_data = data[data['Voltage/kV'] <= threshold]
            exp_data = data[data['Voltage/kV'] > threshold]

            if len(linear_data) < 2 or len(exp_data) < 2:
                # Not enough points to fit
                #Do not attempt to fit the data if there are too few data points given in certain regimes given the thresholds.

                continue

            # Fit linear part
            linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
            linear_fit_func = np.poly1d(linear_fit_params)

            # Fit exponential part
            try:
                c_fixed = threshold
                exp_fit_params, _ = curve_fit(lambda x, a, b: exp_func(x,a,b,c_fixed), exp_data['Voltage/kV'], exp_data['Current/uA'])

                exp_fit_params= np.append(exp_fit_params,c_fixed)
            except RuntimeError:
                # Fit failed
                continue

            # Calculate combined error (sum of squared residuals)
            linear_resid = linear_data['Current/uA'] - linear_fit_func(linear_data['Voltage/kV'])
            SSE= np.sum(linear_resid**2)

            #Calculate R2

            diff = linear_data['Voltage/kV']-linear_data['Voltage/kV'].mean()
            square_diff = diff**2
            SST = np.sum(square_diff)

            R2 = 1 - SSE/SST

            if R2 < 0.99:
                    if R2 > best_R2:
                        best_R2 = R2
                        best_threshold = threshold

        return best_threshold

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
