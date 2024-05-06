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

def chopper(df, n):
    linear = df[df['I/uA'] <= n]
    exp = df[df['I/uA'] >= n]
    return linear, exp

folder_path = "RPC_Efficiencies//IV"

files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

#tube colours
colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

#for i in range(len(files)):
 
i=12
data_path = os.path.join(folder_path, files[i])
df = pd.read_csv(data_path)
if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
    df = df.sort_values(by=['V/kV'], ascending=True)

    #number of points tried
    N = 50

    values = np.linspace(0.031,0.201,N)

    #initialise array for calculating residuals
    residuals_sum = np.zeros(N)

    for j in range(N):   

        #linear and exponential fit
        linear, exp = chopper(df, values[j])
        poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
        a_optLinear, b_optLinear = poptLinear

        #print(files[i])
        #print(values[j])
        #print(np.diag(pcovLinear))

        x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)
        y_linear = fitLinear(x_linear, a_optLinear, b_optLinear)
    

        poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.5], maxfev=2000)
        a_optExp, b_optExp, c_optExp = poptExp

        #print(np.diag(pcovExp))

        x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)
        y_exp = fitExp(x_exp, a_optExp, b_optExp, c_optExp)

        #calculate residuals
        for row in linear.index:
            residuals_sum[j] += ((linear['I/uA'][row]-fitLinear(linear['V/kV'][row], a_optLinear, b_optLinear))/(linear['Uncertainty/uA'][row] if linear['Uncertainty/uA'][row] > 0 else 0.01))**2
        for row in exp.index:
            residuals_sum[j] += ((exp['I/uA'][row]-fitExp(exp['V/kV'][row], a_optExp, b_optExp, c_optExp))/exp['Uncertainty/uA'][row])**2
        print('Cutoff = ' + str(values[j]))
        print('Sum of residuals squared = ' + str(residuals_sum[j]))    

    print(values)
    print(residuals_sum)
    plt.plot(values, residuals_sum)
    plt.show()

    index = np.argmin(residuals_sum)
    print(residuals_sum[index])

    #linear and exponential fit
    linear, exp = chopper(df, values[index])
    poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
    a_optLinear, b_optLinear = poptLinear


    x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)
    y_linear = fitLinear(x_linear, a_optLinear, b_optLinear)


    poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.5], maxfev=2000)
    a_optExp, b_optExp, c_optExp = poptExp

    x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)
    y_exp = fitExp(x_exp, a_optExp, b_optExp, c_optExp)
        
    plt.plot(df['V/kV'], df['I/uA'], marker='.', color=colors[i], label=files[i].split(".")[0], linestyle='None')
    plt.errorbar(df['V/kV'], df['I/uA'], yerr=df['Uncertainty/uA'], fmt='o', capsize=5, label='_nolegend_', color = colors[i])
    plt.plot(x_linear, y_linear, label=f'Linear fit: y = {a_optLinear:.2f}x + {b_optLinear:.2f}', color=colors[i])
    plt.plot(x_exp, y_exp, label=f'Exp fit: y = {b_optExp:.2e} * exp({c_optExp:.2f}x) + {a_optExp:.2f}', color=colors[i])
    plt.xlabel('$V/\mathrm{kV}$')
    plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
    plt.ylim(0)
    title = str(files[i])
    plt.title(title)
    plt.legend()

    #plt.savefig(title + '.png')
    plt.show()

    
    