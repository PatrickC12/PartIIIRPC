import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# folder_path = 'C:\\Users\\Peter\\OneDrive - University of Cambridge\\Desktop\\Project Excel Work\\MiniRPC16'

# Setting the Seaborn theme
sns.set_theme(style="darkgrid")

class CSVPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Plotter")

        # Frame for file selection
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Button to load folder
        self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=5)
        
        # Initialize the plot button here but disabled, ensuring it's always in a valid state when accessed
        self.plot_button = ttk.Button(self.frame, text="Plot Selected", state='disabled', command=self.plot_selected)
        self.plot_button.pack(pady=5)

        # Dictionary to hold the checkboxes and file paths
        self.csv_files = {}
        
        # Plot button, disabled initially
        self.plot_button = ttk.Button(self.frame, text="Plot Selected", state='disabled', command=self.plot_selected)
        self.plot_button.pack(pady=5)

    def load_folder(self):
        global folder_path
        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        if not folder_path:
            # Immediately return if no folder is selected to avoid any further execution
            return
        
        # Clearing previous selections - safely clear widgets
        for widget in self.frame.winfo_children():
            if widget != self.load_button and widget != self.plot_button:
                widget.destroy()

        self.csv_files = {}  # Resetting the csv_files dictionary

        files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        if not files:
            messagebox.showinfo("No CSV Files", "No CSV files found in the selected folder.")
            return

        self.plot_button['state'] = 'normal'  # Safely update the button state after ensuring it's in a valid context

        for file in files:
            self.csv_files[file] = tk.BooleanVar()
            chk = ttk.Checkbutton(self.frame, text=file, variable=self.csv_files[file])
            chk.pack(anchor='w')

    def plot_selected(self):
        # Filter selected files
        selected_files = [file for file, var in self.csv_files.items() if var.get()]

        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return
        
        plt.figure(figsize=(10, 8))

        for file in selected_files:
            data_path = os.path.join(folder_path, file)  # Update 'folder_path' dynamically if needed
            data = pd.read_csv(data_path)
            
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
