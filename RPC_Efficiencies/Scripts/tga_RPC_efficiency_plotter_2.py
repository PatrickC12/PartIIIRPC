import warnings
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

hep.style.use(hep.style.ATLAS)

def fitLinear(x, a, b):
    return a*x + b

def fitExp(x,a,b,c):
    return a + b*np.exp(c*x)

def chopper(df):
    linear = df[df['I/uA'] <= 0.1]
    exp = df[df['I/uA'] >= 0.05]
    return linear, exp

class CSVPlotterApp:
    def __init__(self, master):
        self.master = master
        master.title("RPC Efficiency Plotter")
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.load_button = ttk.Button(self.frame, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=5)

        self.csv_files = {}

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

            #choose your option (only pick one)
            self.plot_iv_button = ttk.Button(self.frame, text="IV curve", command=lambda: self.plot_selected_iv(folder_path))
            self.plot_iv_button.pack(pady=5)

            self.plot_hv_button = ttk.Button(self.frame, text="HV efficiency", command=lambda: self.plot_selected_hv(folder_path))
            self.plot_hv_button.pack(pady=5)

            self.plot_vth_button = ttk.Button(self.frame, text="Vth efficiency", command=lambda: self.plot_selected_vth(folder_path))
            self.plot_vth_button.pack(pady=5)

            self.plot_contour_button = ttk.Button(self.frame, text="Contour plot", command=lambda: self.plot_selected_contour(folder_path))
            self.plot_contour_button.pack(pady=5)

            #additional option
            self.separate_var = tk.BooleanVar()
            self.separate_chk = ttk.Checkbutton(self.frame, text="Plot separately", variable=self.separate_var)
            self.separate_chk.pack(pady=5)

    def plot_selected_iv(self, folder_path):
        if self.separate_var.get():
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return
            
            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
                    df = df.sort_values(by=['V/kV'], ascending=True)

                    plt.plot(df['V/kV'], df['I/uA'], marker='.', color=colors[i], label=selected_files[i].split(".")[0])
                    plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='o', capsize=5, label='_nolegend_', color = colors[i])

                    #linear and exponential fit
                    linear, exp = chopper(df)
                    poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
                    a_optLinear, b_optLinear = poptLinear
                    x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)
                    y_linear = fitLinear(x_linear, a_optLinear, b_optLinear)
                    plt.plot(x_linear, y_linear)

                    poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.5], maxfev=2000)
                    a_optExp, b_optExp, c_optExp = poptExp

                    x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)
                    y_exp = fitExp(x_exp, a_optExp, b_optExp, c_optExp)
                    plt.plot(x_exp, y_exp)

                    plt.xlabel('$V/\mathrm{kV}$')
                    plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
                    plt.ylim(0)
                    title = str(selected_files[i])
                    plt.title(title)
                    plt.legend()
                    #plt.savefig(title + '.png')
                    plt.show()

        else:
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return
            
            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
                    df = df.sort_values(by=['V/kV'], ascending=True)
                    plt.plot(df['V/kV'], df['I/uA'], marker='.', color=colors[i], label=selected_files[i].split(".")[0])
                    plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='o', capsize=5, label='_nolegend_', color = colors[i])


            plt.xlabel('$V/\mathrm{kV}$')
            plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
            plt.ylim(0)
            #title = str(selected_files[i])
            plt.title('Leakage current against HV')
            plt.legend()
            #plt.savefig(title + '.png')
            plt.show()

    def plot_selected_hv(self, folder_path):
        if self.separate_var.get():
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return

            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'Numerator' in df.columns:
                    df['Efficiency/%'] = df['Numerator']/df['Denominator']*100
                    df = df.sort_values(by=['HV/kV'], ascending=True)
                    plt.plot(df['HV/kV'], df['Efficiency/%'], marker='.',color=colors[i])
                    errors=np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator']))
                    plt.errorbar(df['HV/kV'], df['Efficiency/%'], yerr=errors*100, fmt='o', capsize=5, label='_nolegend_', color = colors[i])
                    
                    plt.xlabel('$V/\mathrm{kV}$')
                    plt.ylabel('Efficiency/%')
                    plt.ylim(0)
                    title = str(selected_files[i])
                    plt.title(title)
                    #plt.savefig(title + '.Efficiency.png')
                    plt.show()
        else:
            plt.figure()
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return

            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'Numerator' in df.columns:
                    df['Efficiency/%'] = df['Numerator']/df['Denominator']*100
                    df = df.sort_values(by=['HV/kV'], ascending=True)
                    plt.plot(df['HV/kV'], df['Efficiency/%'], marker='.',color=colors[i], label=selected_files[i].split(".")[0])
                    errors=np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator']))
                    plt.errorbar(df['HV/kV'], df['Efficiency/%'], yerr=errors*100, fmt='o', capsize=5, label='_nolegend_', color = colors[i])
                    
            plt.xlabel('$V/\mathrm{kV}$')
            plt.ylabel('Efficiency/%')
            plt.ylim(0)
            #title = str(selected_files[i])
            plt.title("Efficiency against HV")
            plt.legend()
            #plt.savefig(title + '.Efficiency.png')
            plt.show()      

    def plot_selected_vth(self, folder_path):
        if self.separate_var.get():
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return

            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'Numerator' in df.columns:
                    plt.plot(df['Vth/V'], df['Efficiency/%'], marker='.',color=colors[i])
                    errors=np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator']))
                    plt.errorbar(df['Vth/V'], df['Efficiency/%'], yerr=errors*100, fmt='o', capsize=5, label='_nolegend_', color = colors[i])
                    plt.xlabel('$V_{\mathrm{th}}/\mathrm{V}$')
                    plt.ylabel('Efficiency/%')
                    plt.title(selected_files[i])
                    plt.ylim(0)
                    #plt.savefig(title + '.Efficiency.png')
                    plt.show()
        else:
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return

            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'Numerator' in df.columns:
                    plt.plot(df['Vth/V'], df['Efficiency/%'], marker='.',color=colors[i], label=selected_files[i].split(".")[0])
                    errors=np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator']))
                    plt.errorbar(df['Vth/V'], df['Efficiency/%'], yerr=errors*100, fmt='o', capsize=5, label='_nolegend_', color = colors[i])
            plt.xlabel('$V_{\mathrm{th}}/\mathrm{V}$')
            plt.ylabel('Efficiency/%')
            plt.title("Efficiency against $V_{\mathrm{th}}$")
            plt.ylim(0)
            plt.legend()
            #plt.savefig(title + '.Efficiency.png')
            plt.show()
            
    def plot_selected_contour(self, folder_path):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return

        #tube colours
        colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

        for i in range(len(selected_files)):
            data_path = os.path.join(folder_path, selected_files[i])
            df = pd.read_csv(data_path)

            # Make data
            X = np.array(df['Vth/V'])
            Y = np.array(df['HV/kV'])
            Z = np.array(df['Efficiency/%'])

            # Define a regular grid
            xi = np.linspace(X.min(), X.max(), 100)
            yi = np.linspace(Y.min(), Y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate the data onto the regular grid
            zi = griddata((X, Y), Z, (xi, yi), method='linear')

            # Create the surface plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xi, yi, zi, cmap='BuPu')

            # Add labels and title
            ax.set_xlabel('$V_{\mathrm{th}}$/V')
            ax.set_ylabel('HV/kV')
            ax.set_zlabel('Efficiency/%')
            ax.set_title('Pure Freon Efficiency vs $V_{\mathrm{th}}$ and HV')

            #pad so that colour bar does not end up on top of z axis
            fig.colorbar(surf, pad=0.12)

            plt.title(selected_files[i])
            #plt.savefig(title + '3D contour.png')

            # Show the plot
            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()