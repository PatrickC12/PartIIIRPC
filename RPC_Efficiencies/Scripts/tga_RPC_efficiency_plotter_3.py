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

def fitCombined(x,a,b,c):
    return a*x + b*np.exp(c*x)

def chopper(df, n):
    linear = df[df['I/uA'] <= n]
    exp = df[df['I/uA'] >= n]
    return linear, exp

def finalChop(df, threshold):
    linear = df.iloc[:threshold+1]
    exp = df.iloc[threshold:]
    return linear, exp

def voltage1uA(poptExp):
    return 1/poptExp[2] * np.log((1-poptExp[0])/poptExp[1])

def critVoltage(linear, df, poptLinear, poptExp):
    trials = np.linspace(linear['V/kV'].max(), df['V/kV'].max(), 100)
    for i in trials:
        if fitLinear(i, *poptLinear) * 2 < fitExp(i, *poptExp):
            return i, fitExp(i, *poptExp)
        
def fittingPoint(df):
    df = df.reset_index(drop=True)

    #number of points tried
    N = 50

    #values = np.linspace(0.031,0.201,N)
    values = df['I/uA'][(df['I/uA'] >= 0.03) & (df['I/uA'] <= 0.20)]
    N = len(values)
    
    #initialise array for calculating residuals
    residuals_sum = np.zeros(N)

    for j in range(N):   

        #linear and exponential fit
        #print(values.iloc[j])
        linear, exp = chopper(df, values.iloc[j])
        poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[0.01,0])
        a_optLinear, b_optLinear = poptLinear

        x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)

        poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.001,3], maxfev=5000)
        a_optExp, b_optExp, c_optExp = poptExp

        x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)

        #calculate residuals
        for row in linear.index:
            residuals_sum[j] += (linear['I/uA'][row]-fitLinear(linear['V/kV'][row], a_optLinear, b_optLinear))**2
        for row in exp.index:
            residuals_sum[j] += (exp['I/uA'][row]-fitExp(exp['V/kV'][row], a_optExp, b_optExp, c_optExp))**2

    index = np.argmin(residuals_sum)
    
    threshold = values.index[index]
    
    #linear and exponential fit
    linear, exp = finalChop(df, threshold)
    poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
    a_optLinear, b_optLinear = poptLinear

    poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.001,3], maxfev=4000)
    a_optExp, b_optExp, c_optExp = poptExp

    #combined fit (new for residuals_2)
    poptCombined, pcovCombined = curve_fit(fitCombined, df['V/kV'], df['I/uA'], p0=[a_optLinear, b_optExp, c_optExp])
    a_optCombined, b_optCombined, c_optCombined = poptCombined

    x_combined = np.linspace(df['V/kV'].min(), df['V/kV'].max(), 100)
    y_combined = fitCombined(x_combined, a_optCombined, b_optCombined, c_optCombined)

    return critVoltage(linear, df, poptLinear, poptExp), poptCombined, x_combined, y_combined, poptExp

def critical_voltages(gas, voltage, current, poptExp):

    data_path = "C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/RPC_Efficiencies/Critical_voltages.csv"
    df = pd.read_csv(data_path)

    v1uA = voltage1uA(poptExp)

    dfNew = pd.DataFrame([[gas, voltage, current, v1uA, 'blank', 'blank']], columns=['File','Critical voltage/kV','Critical current/uA', 'V(1uA)/kV', 'Max_eff/%', 'plateau/%'])
    df = pd.concat([df, dfNew])

    df.to_csv('Critical_voltages.csv')

    return 

def fittingPoint_large_rpc(df):
    df = df.reset_index(drop=True)

    #number of points tried
    N = 50

    #values = np.linspace(0.031,0.201,N)
    values = df['Current/uA'][(df['Current/uA'] >= 0.03) & (df['Current/uA'] <= 10.0)]
    N = len(values)
    
    #initialise array for calculating residuals
    residuals_sum = np.zeros(N)

    for j in range(N):   

        #linear and exponential fit
        #print(values.iloc[j])
        linear, exp = chopper(df, values.iloc[j])
        poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
        a_optLinear, b_optLinear = poptLinear

        x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)

        poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.7], maxfev=5000)
        a_optExp, b_optExp, c_optExp = poptExp

        x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)

        #calculate residuals
        for row in linear.index:
            residuals_sum[j] += (linear['I/uA'][row]-fitLinear(linear['V/kV'][row], a_optLinear, b_optLinear))**2
        for row in exp.index:
            residuals_sum[j] += (exp['I/uA'][row]-fitExp(exp['V/kV'][row], a_optExp, b_optExp, c_optExp))**2

    index = np.argmin(residuals_sum)
    
    threshold = values.index[index]
    
    #linear and exponential fit
    linear, exp = finalChop(df, threshold)
    poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
    a_optLinear, b_optLinear = poptLinear

    poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.7], maxfev=4000)
    a_optExp, b_optExp, c_optExp = poptExp

    #combined fit (new for residuals_2)
    poptCombined, pcovCombined = curve_fit(fitCombined, df['V/kV'], df['I/uA'], p0=[a_optLinear, b_optExp, c_optExp])
    a_optCombined, b_optCombined, c_optCombined = poptCombined

    x_combined = np.linspace(df['V/kV'].min(), df['V/kV'].max(), 100)
    y_combined = fitCombined(x_combined, a_optCombined, b_optCombined, c_optCombined)

    return critVoltage(linear, df, poptLinear, poptExp), poptCombined, x_combined, y_combined

def efficiency_fit_polynom(df):
    x=df[df['HV/kV']>0]['HV/kV'].astype(float)
    y=df[df['Efficiency/%']>0]['Efficiency/%'].astype(float)

    p= np.polyfit(x, y, 5)
    x_values = np.linspace(df['HV/kV'].min(), df['HV/kV'].max(), 100)
    y_values = p[0] * x_values**5 + p[1] * x_values**4 + p[2] * x_values**3 + p[3] * x_values**2 + p[4] * x_values + p[5]

    return x_values, y_values

def logistic(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))

def efficiency_fit_logistic(df):
    x=df[df['Efficiency/%']>0]['HV/kV'].astype(float)
    y=df[df['Efficiency/%']>0]['Efficiency/%'].astype(float)

    poptLogistic, pcovLogistic = curve_fit(logistic, x, y, p0=[80, 7, 6])

    print(poptLogistic)

    x_values = np.linspace(df['HV/kV'].min(), df['HV/kV'].max(), 100)
    y_values = logistic(x_values, *poptLogistic)

    return x_values, y_values, poptLogistic

def peak_efficiencies(gas, file, poptLogistic):
    data_path = "C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/RPC_Efficiencies/Critical_voltages.csv"
    df = pd.read_csv(data_path)

    maxEff = file['Efficiency/%'].max()
    plateau = poptLogistic[0]
    error = file[file['Efficiency/%'] == maxEff]['error'].values[0]

    dfNew = pd.DataFrame([[gas, 'blank', 'blank', 'blank', maxEff, plateau, error]], columns=['File','Critical voltage/kV','Critical current/uA', 'V(1uA)/kV', 'Max_eff/%', 'plateau/%', 'error_in_max/%'])
    df = pd.concat([df, dfNew])

    df.to_csv('Critical_voltages.csv')

    return 


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
            self.plot_data_button = ttk.Button(self.frame, text="Data", command=lambda: self.data(folder_path))
            self.plot_data_button.pack(pady=5)

            self.plot_data_hv_button = ttk.Button(self.frame, text="Other data iykyk", command=lambda: self.data_hv(folder_path))
            self.plot_data_hv_button.pack(pady=5)

            self.plot_iv_button = ttk.Button(self.frame, text="IV curve", command=lambda: self.plot_selected_iv(folder_path))
            self.plot_iv_button.pack(pady=5)

            self.plot_iv_large_rpc_button = ttk.Button(self.frame, text="Large RPC IV curve", command=lambda: self.plot_selected_iv_large_rpc(folder_path))
            self.plot_iv_large_rpc_button.pack(pady=5)

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

            self.fitting_var = tk.BooleanVar()
            self.fitting_chk = ttk.Checkbutton(self.frame, text="Find V_crit", variable=self.fitting_var)
            self.fitting_chk.pack(pady=5)

    def data(self, folder_path):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return
        
        for i in range(len(selected_files)):
            data_path = os.path.join(folder_path, selected_files[i])
            df = pd.read_csv(data_path)
            if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
                df = df.sort_values(by=['V/kV'], ascending=True)
                #fitting function
                voltageCurrent, poptCombined, x_combined, y_combined, poptExp = fittingPoint(df)
                voltage, current = voltageCurrent

                #read critical voltages
                critical_voltages(selected_files[i], voltage, current, poptExp)
        
        messagebox.showinfo("Message", "Data reading complete")

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

                    #fitting function
                    voltageCurrent, poptCombined, x_combined, y_combined, poptExp = fittingPoint(df)
                    voltage, current = voltageCurrent
                    plt.plot(x_combined, y_combined, color=colors[i], label = 'best fit curve $I={0:.2f}V+{1:.10f} \exp {2:.2f}V$'.format(*poptCombined))
                    
                    if self.fitting_var.get():
                        voltageStr = str(round(voltage, 2))
                        plt.scatter(voltage, current, color='r', marker='x', s=100, zorder=10, label = "Critical point = " + voltageStr + '$\mathrm{kV}$')
                        print(voltage, current)

                    plt.plot(df['V/kV'], df['I/uA'], marker='.', color=colors[i], linestyle='None', label=selected_files[i].split(".")[0])
                    plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='o', capsize=5, label='_nolegend_', color = colors[i])

                    plt.xlabel('$V/\mathrm{kV}$')
                    plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
                    plt.ylim(0)
                    title = str(selected_files[i])
                    plt.title(title)
                    plt.legend()
                    #plt.savefig(title + '.png')
                    plt.show()
                else:
                    messagebox.showwarning("Wrong folder!!")
                

        else:
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return
            
            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]
            markers = ['^', 'v', 'D', 'd', 'x', 'o']
            captions=['94.7% Freon, 5% Isobutane, 0.3% $\mathrm{SF_6}$', '95% Freon, 5% Isobutane', '80% CERN, 20% $\mathrm{CO_2}$', '70% CERN, 30% $\mathrm{CO_2}$', '60% CERN, 40% $\mathrm{CO_2}$']
            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
                    df = df.sort_values(by=['V/kV'], ascending=True)
                    #fitting function
                    voltageCurrent, poptCombined, x_combined, y_combined, poptExp = fittingPoint(df)
                    voltage, current = voltageCurrent

                    #read critical voltages
                    #critical_voltages(selected_files[i], voltage, current, poptExp)

                    plt.plot(x_combined, y_combined, color=colors[i])
                    if self.fitting_var.get():
                        voltageStr = str(round(voltage, 2))
                        plt.scatter(voltage, current, color='r', marker='x', s=100, zorder=10, label = "Critical point = " + voltageStr + '$\mathrm{kV}$')
                    plt.plot(df['V/kV'], df['I/uA'], marker=markers[i], color=colors[i], linestyle='None', label=captions[i])
                    plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='None', capsize=5, label='_nolegend_', color = colors[i])
                    
                else:
                    messagebox.showwarning("Wrong folder!!")
            plt.xlabel('$V$ $[\mathrm{kV}$]')
            plt.ylabel('$I$ $[\mathrm{\mu A}$]')
            plt.ylim(0)
            #title = str(selected_files[i])
            title = "Adding $\mathrm{SF_6}$"
            plt.title(title)
            plt.legend()
            #plt.figure(figsize=[6.4, 4.8])
            #plt.savefig(title + '.pdf')
            plt.show()

    def plot_selected_iv_large_rpc(self, folder_path):
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return
        
        #tube colours
        colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

        for i in range(len(selected_files)):
            data_path = os.path.join(folder_path, selected_files[i])
            df = pd.read_csv(data_path)
            df['I/uA'] = df['Current/uA']
            df['V/kV'] = df['Voltage/kV']
            if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
                df = df.sort_values(by=['V/kV'], ascending=True)
                #fitting function
                voltageCurrent, poptCombined, x_combined, y_combined = fittingPoint_large_rpc(df)
                voltage, current = voltageCurrent
                plt.plot(x_combined, y_combined, color=colors[i])
                if self.fitting_var.get():
                    voltageStr = str(round(voltage, 2))
                    plt.scatter(voltage, current, color='r', marker='x', s=100, zorder=10, label = "Critical point = " + voltageStr + '$\mathrm{kV}$')
                plt.plot(df['V/kV'], df['I/uA'], marker='.', color=colors[i], linestyle='None', label=selected_files[i].split(".")[0])
                plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='o', capsize=5, label='_nolegend_', color = colors[i])
                
            else:
                messagebox.showwarning("Wrong folder!!")
        plt.xlabel('$V/\mathrm{kV}$')
        plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
        plt.ylim(0)
        #title = str(selected_files[i])
        title = str(selected_files[i])
        plt.title(title)
        plt.legend()
        #plt.savefig(title + '.pdf')
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
                    plt.savefig(title + '.Efficiency.png')
                    plt.show()
        else:
            plt.figure()
            selected_files = [file for file, var in self.csv_files.items() if var.get()]
            if not selected_files:
                messagebox.showwarning("No Selection", "No files selected for plotting.")
                return

            #tube colours
            colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]
            markers = ['^', 'v', 'D', 'd', 'x', 'o']
            captions=['94.7% Freon, 5% Isobutane, 0.3% $\mathrm{SF_6}$', '95% Freon, 5% Isobutane', '80% CERN, 20% $\mathrm{CO_2}$', '70% CERN, 30% $\mathrm{CO_2}$', '60% CERN, 40% $\mathrm{CO_2}$']
            
            for i in range(len(selected_files)):
                data_path = os.path.join(folder_path, selected_files[i])
                df = pd.read_csv(data_path)
                if 'Numerator' in df.columns:
                    df['Efficiency/%'] = df['Numerator']/df['Denominator']*100
                    df = df.sort_values(by=['HV/kV'], ascending=True)
                    maxEff = df['Efficiency/%'].max()

                    #plot polynomial fit
                    #x_values, y_values = efficiency_fit_polynom(df)
                    #plt.plot(x_values, y_values, label="polyfit")

                    #plot alternative fit
                    x_values, y_values, poptLogistic = efficiency_fit_logistic(df)

                    #plot logistic fit
                    plt.plot(x_values, y_values, label='_nolegend_', color=colors[i])

                    plt.plot(df['HV/kV'], df['Efficiency/%'], marker=markers[i],color=colors[i], linestyle='None', label=captions[i])
                    errors=np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator']))
                    plt.errorbar(df['HV/kV'], df['Efficiency/%'], yerr=errors*100, fmt='None', capsize=5, label='_nolegend_', color = colors[i])
                    plt.plot(x_values, y_values, label='_nolegend_', color=colors[i])
                    
            plt.xlabel('$V$ [$\mathrm{kV}$]')
            plt.ylabel('Efficiency [%]')
            plt.ylim(0)
            #title = str(selected_files[i])
            plt.title("Adding $\mathrm{SF_6}$")
            plt.legend()
            #plt.savefig(title + '.Efficiency.png')
            plt.show()  

    def data_hv(self, folder_path):
        plt.figure()
        selected_files = [file for file, var in self.csv_files.items() if var.get()]
        if not selected_files:
            messagebox.showwarning("No Selection", "No files selected for plotting.")
            return

        for i in range(len(selected_files)):
            data_path = os.path.join(folder_path, selected_files[i])
            df = pd.read_csv(data_path)
            if 'Numerator' in df.columns:
                df['Efficiency/%'] = df['Numerator']/df['Denominator']*100
                df = df.sort_values(by=['HV/kV'], ascending=True)
                df['error']=(np.sqrt((df['Efficiency/%']/100*(1-df['Efficiency/%']/100)/df['Denominator'])))*100

                #plot alternative fit
                x_values, y_values, poptLogistic = efficiency_fit_logistic(df)

                #take max efficiencies
                peak_efficiencies(selected_files[i], df, poptLogistic) 

        messagebox.showinfo("Message", "Data reading complete")   

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

            for i in range(5):
                vthRange=[1.7, 1.8,1.9,2.0,2.1]
                d = df.loc[df["Vth/V"] == vthRange[i]]
                lab='$V_{\mathrm{th}} =$ ' + str(vthRange[i]) + ' V'
                plt.plot(d['HV/kV'],d['Efficiency/%'], marker='.', label = lab, color=colors[i])
                errors=np.sqrt((d['Efficiency/%']/100*(1-d['Efficiency/%']/100)/d['Denominator']))
                plt.errorbar(d['HV/kV'], d['Efficiency/%'], yerr=errors*100, fmt='o', capsize=5, label='_nolegend_', color = colors[i])
            plt.title('I vs HV. Pure Freon')
            plt.xlabel('HV/kV')
            plt.ylabel('Efficiency/%')
            plt.legend()

            plt.savefig('contour plot pure freon')
            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()