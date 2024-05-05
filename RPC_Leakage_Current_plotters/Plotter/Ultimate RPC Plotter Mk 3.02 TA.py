import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import os
from matplotlib.pyplot import cm
import mplhep as hep

hep.style.use(hep.style.ATLAS) 

# Setting the Seaborn theme
#sns.set_theme(style="darkgrid")

# Define the fitting function for exponential fitting
def exp_func(x, a, b, c):
    try:
        return a * np.exp(b * (x - c))
    except OverflowError:
        return np.inf

class CSVPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Plotter")
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=5)

        self.csv_files = {}

    def find_best_threshold(self, data):
        best_R2 = -np.inf
        best_threshold = None

        voltage = data['Voltage/kV'].values
        total_points = len(voltage)
        thresholds = np.linspace(voltage.min(), voltage.max(), 100)

        for threshold in thresholds[:-1]:  # Exclude the last point
            mask_linear = voltage <= threshold
            mask_exp = voltage > threshold

            # Calculate the percentage of points in each region
            percent_linear = sum(mask_linear) / total_points * 100
            percent_exp = sum(mask_exp) / total_points * 100

            # Continue if either region has too few points (e.g., less than 10% of total points)
            if percent_linear < 25 or percent_exp < 10:
                continue

            # The scoring could be refined to consider the quality of fit, but for now, we use the sum of percentages
            current_score = percent_linear + percent_exp
            if current_score > best_R2:
                best_R2 = current_score
                best_threshold = threshold

        return best_threshold

    def load_folder(self):
            folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
            if not folder_path:
                return

            for widget in self.frame.winfo_children():
                widget.destroy()

            self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
            self.load_button.pack(pady=5)

            self.csv_files = {}
            files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
            if not files:
                messagebox.showinfo("No CSV Files", "No CSV files found in the selected folder.")
                return

            for file in files:
                self.csv_files[file] = tk.BooleanVar()
                chk = ttk.Checkbutton(self.frame, text=file, variable=self.csv_files[file])
                chk.pack(anchor='w')

            self.plot_button = ttk.Button(self.frame, text="Plot Selected", command=lambda: self.plot_selected(folder_path))
            self.plot_button.pack(pady=5)

            # Additional UI elements for manual threshold and initial guess values
            self.manual_threshold_var = tk.DoubleVar()
            self.manual_threshold_entry = ttk.Entry(self.frame, textvariable=self.manual_threshold_var)
            self.manual_threshold_entry.pack(pady=5)
            self.manual_threshold_label = ttk.Label(self.frame, text="Manual Threshold Voltage (kV):")
            self.manual_threshold_label.pack(pady=5)

            self.initial_guess_a_var = tk.DoubleVar(value=1.0)
            self.initial_guess_b_var = tk.DoubleVar(value=0.1)
            self.initial_guess_c_var = tk.DoubleVar(value=0.0)
            self.initial_guess_entries = {
                'a': ttk.Entry(self.frame, textvariable=self.initial_guess_a_var),
                'b': ttk.Entry(self.frame, textvariable=self.initial_guess_b_var),
                'c': ttk.Entry(self.frame, textvariable=self.initial_guess_c_var)
            }
            for param, entry in self.initial_guess_entries.items():
                entry.pack(pady=2)
                label = ttk.Label(self.frame, text=f"Initial Guess {param}:")
                label.pack(pady=2)

            # Re-adding checkboxes for additional options after folder is selected
            self.error_bar_var = tk.BooleanVar()
            self.error_bar_chk = ttk.Checkbutton(self.frame, text="Add Error Bars", variable=self.error_bar_var)
            self.error_bar_chk.pack(pady=5)

            self.fit_lines_var = tk.BooleanVar()
            self.fit_lines_chk = ttk.Checkbutton(self.frame, text="Fit Linear and Exponential Regions", variable=self.fit_lines_var)
            self.fit_lines_chk.pack(pady=5)

    def plot_selected(self, folder_path):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return

        plt.figure(figsize=(20, 16))
        manual_threshold = self.manual_threshold_var.get()
        initial_guesses = [self.initial_guess_a_var.get(), self.initial_guess_b_var.get(), self.initial_guess_c_var.get()]

        color_map = plt.cm.get_cmap('rainbow', len(selected_files))
        for file, color_index in zip(selected_files, range(len(selected_files))):
            data_path = os.path.join(folder_path, file)
            data = pd.read_csv(data_path)
            data = data[data['Current/uA'] > 0]
            data = data.sort_values(by=['Voltage/kV'], ascending=True)

            #Debugging: check shape of data inputted into sns.lineplot
            #print(data['Voltage/kV'].shape)
            #print(data['Current/uA'].shape)

            #print("File:", file)
            #print("Data:", data)

            #colors = plt.cm.rainbow(np.linspace(0, 1.0, len(selected_files)))  # Generate a list of colors.
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]
            color_index = selected_files.index(file)  # Get the index of the current file
            color = colors[color_index]  # Select the color for the current file
            if color_index % 2 == 0:
                #sns.lineplot(x=data['Voltage/kV'], y=data['Current/uA'], label=file.split(".")[0],marker='o', markersize=5, color=color)
                plt.plot(data['Voltage/kV'], data['Current/uA'], label=file.split(".")[0],marker='o',markersize=5, color=color)
            else:
                #sns.lineplot(x=data['Voltage/kV'], y=data['Current/uA'], label=file.split(".")[0],marker='o', linestyle = '--', markersize=5, color=color) 
                plt.plot(data['Voltage/kV'], data['Current/uA'], label=file.split(".")[0],marker='o',markersize=5, color=color)
            #sns.scatterplot(x=data['Voltage/kV'], y=data['Current/uA'], color=color) 


            if self.fit_lines_var.get():
                if manual_threshold > 0:
                    threshold = manual_threshold
                else:
                    threshold = self.find_best_threshold(data[['Voltage/kV', 'Current/uA']])

                if threshold is not None:
                    linear_data = data[data['Voltage/kV'] <= threshold]
                    if len(linear_data) > 1:
                        linear_fit_params = np.polyfit(linear_data['Voltage/kV'], linear_data['Current/uA'], 1)
                        linear_fit_func = np.poly1d(linear_fit_params)
                        linear_x_vals = np.linspace(data['Voltage/kV'].min(), threshold, 100)
                        label_linear_fit = f'Linear Fit: $y = {linear_fit_params[0]:.2f}x + {linear_fit_params[1]:.2f}$'
                        plt.plot(linear_x_vals, linear_fit_func(linear_x_vals), linestyle="--", label=label_linear_fit)

                    exp_data = data[data['Voltage/kV'] > threshold]
                    if len(exp_data) > 1:
                        try:
                            exp_fit_params, _ = curve_fit(exp_func, exp_data['Voltage/kV'], exp_data['Current/uA'], p0=initial_guesses, maxfev=10000)
                            exp_x_vals = np.linspace(threshold, data['Voltage/kV'].max(), 100)
                            label_exp_fit = f'Exp Fit: $y = {exp_fit_params[0]:.2f} \cdot e^{{{exp_fit_params[1]:.2f}(x-{exp_fit_params[2]:.2f})}}$'
                            plt.plot(exp_x_vals, exp_func(exp_x_vals, *exp_fit_params), linestyle="--", label=label_exp_fit)
                        except RuntimeError as e:
                            messagebox.showwarning("Fit Error", f"Exponential fit failed for {file}: {e}")

            if self.error_bar_var.get():
                errors = data.iloc[:, 2].replace(0, 0.01)
                plt.errorbar(data['Voltage/kV'], data['Current/uA'], yerr=errors, fmt='o', capsize=5, label='_nolegend_', color = color)

        plt.xlabel('Voltage/kV')
        plt.ylim(0)
        plt.ylabel('Current/$\mathrm{\mu} A$')
        plt.title('Threshold Voltage against Current')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()
