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

        # Error bar checkbox
        self.error_bar_var = tk.BooleanVar()

        # Checkbox for using threshold voltage fitting
        self.use_threshold_var = tk.BooleanVar()
        self.use_threshold_chk = ttk.Checkbutton(self.frame, text="Use Threshold Voltage Fitting", variable=self.use_threshold_var, command=self.toggle_threshold_inputs)
        self.use_threshold_chk.pack(pady=5)

        # Initially hide threshold entries, will be toggled by the checkbox
        self.threshold1_label = ttk.Label(self.frame, text="Threshold 1 (kV):")
        self.threshold1_entry = ttk.Entry(self.frame)
        self.threshold2_label = ttk.Label(self.frame, text="Threshold 2 (kV):")
        self.threshold2_entry = ttk.Entry(self.frame)
        
        # Initially disable the entries for threshold values
        self.toggle_threshold_inputs()

    def toggle_threshold_inputs(self):
        if self.use_threshold_var.get():
            # Show and enable threshold entries
            self.threshold1_label.pack(pady=(5, 0))
            self.threshold1_entry.pack(pady=(0, 5))
            self.threshold2_label.pack(pady=(5, 0))
            self.threshold2_entry.pack(pady=(0, 5))
        else:
            # Hide and clear threshold entries
            self.threshold1_label.pack_forget()
            self.threshold1_entry.pack_forget()
            self.threshold2_label.pack_forget()
            self.threshold2_entry.pack_forget()
            self.threshold1_entry.delete(0, tk.END)
            self.threshold2_entry.delete(0, tk.END)

    def load_folder(self):
        global folder_path
        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        if not folder_path:
            return
        
        for widget in self.frame.winfo_children():
            if widget not in [self.load_button, self.plot_button, self.use_threshold_chk, self.error_bar_var]:
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

        # Error bar checkbox creation here
        self.error_bar_chk = ttk.Checkbutton(self.frame, text="Add Error Bars", variable=self.error_bar_var)
        self.error_bar_chk.pack(pady=5)

    def plot_selected(self):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return
        
        plt.figure(figsize=(10, 8))

        for file in selected_files:
            data_path = os.path.join(folder_path, file)
            data = pd.read_csv(data_path)

            if self.use_threshold_var.get():
                try:
                    threshold1 = float(self.threshold1_entry.get())
                    threshold2 = float(self.threshold2_entry.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers for thresholds.")
                    return

                # Plotting with thresholds
                # Add the fitting logic here as before
                
            sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], label=file)

            if self.error_bar_var.get():
                # Adjusting error values: replace 0 with 0.01
                errors = data.iloc[:, 2].replace(0, 0.01)
                plt.errorbar(data.iloc[:, 0], data.iloc[:, 1], yerr=errors, fmt='o', capsize=5, label='_nolegend_')

        plt.xlabel('Voltage (kV)')
        plt.ylabel('Current (uA)')
        plt.title('Voltage vs. Current')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()
