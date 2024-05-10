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
import math

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

folder_path = "C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/RPC_Efficiencies/IV"

files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

#tube colours
colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]

#for i in range(len(files)):
 
i=8
data_path = os.path.join(folder_path, files[i])
df = pd.read_csv(data_path)
if 'I/uA' in df.columns and 'Efficiency/%' not in df.columns:
    df = df.sort_values(by=['V/kV'], ascending=True)
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
        poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
        a_optLinear, b_optLinear = poptLinear

        #print(files[i])
        #print(values[j])
        #print(np.diag(pcovLinear))

        #x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)
        #y_linear = fitLinear(x_linear, a_optLinear, b_optLinear)
    

        poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.5], maxfev=2000)
        a_optExp, b_optExp, c_optExp = poptExp

        #print(np.diag(pcovExp))

        #x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)
        #y_exp = fitExp(x_exp, a_optExp, b_optExp, c_optExp)

        

        #calculate residuals
        for row in linear.index:
            #residuals_sum[j] += ((linear['I/uA'][row]-fitLinear(linear['V/kV'][row], a_optLinear, b_optLinear))/(linear['Uncertainty/uA'][row] if linear['Uncertainty/uA'][row] > 0 else 0.01))**2
            residuals_sum[j] += (linear['I/uA'][row]-fitLinear(linear['V/kV'][row], a_optLinear, b_optLinear))**2
        for row in exp.index:
            #residuals_sum[j] += ((exp['I/uA'][row]-fitExp(exp['V/kV'][row], a_optExp, b_optExp, c_optExp))/exp['Uncertainty/uA'][row])**2
            residuals_sum[j] += (exp['I/uA'][row]-fitExp(exp['V/kV'][row], a_optExp, b_optExp, c_optExp))**2
        #print('Cutoff = ' + str(values.iloc[j]))
        #print('Sum of residuals squared = ' + str(residuals_sum[j]))

    """
    print(values)
    print(residuals_sum)
    plt.plot(values, residuals_sum)
    plt.xlabel('Critical current $\mathrm{I}/\mathrm{\mu A}$')
    plt.ylabel('Residual')
    plt.ylim(0)
    title = str(files[i])
    plt.title(title)
    plt.show()"""

    index = np.argmin(residuals_sum)
    print(residuals_sum[index])
    
    print("index " + str(index))
    threshold = values.index[index]
    print("threshold " + str(threshold))

    #linear and exponential fit
    linear, exp = finalChop(df, threshold)
    poptLinear, pcovLinear = curve_fit(fitLinear, linear['V/kV'], linear['I/uA'], p0=[1,1])
    a_optLinear, b_optLinear = poptLinear

    x_linear = np.linspace(linear['V/kV'].min(), linear['V/kV'].max(), 100)
    y_linear = fitLinear(x_linear, a_optLinear, b_optLinear)


    poptExp, pcovExp = curve_fit(fitExp, exp['V/kV'], exp['I/uA'], p0=[0,0.1,0.5], maxfev=2000)
    a_optExp, b_optExp, c_optExp = poptExp

    x_exp = np.linspace(exp['V/kV'].min(), exp['V/kV'].max(), 100)
    y_exp = fitExp(x_exp, a_optExp, b_optExp, c_optExp)

    #combined fit (new for residuals_2)
    poptCombined, pcovCombined = curve_fit(fitCombined, df['V/kV'], df['I/uA'], p0=[a_optLinear, b_optExp, c_optExp])
    a_optCombined, b_optCombined, c_optCombined = poptCombined

    x_combined = np.linspace(df['V/kV'].min(), df['V/kV'].max(), 100)
    y_combined = fitCombined(x_combined, a_optCombined, b_optCombined, c_optCombined)

    print('critV')
    print(critVoltage(linear, df, poptLinear, poptExp))
        
    plt.plot(df['V/kV'], df['I/uA'], color=colors[i], label=files[i].split(".")[0], linestyle='None')
    plt.errorbar(df['V/kV'], df['I/uA'], fmt='o', yerr=df['Uncertainty/uA'], capsize=5, label='_nolegend_', color = colors[i])
    #labelLinear = f'$I = {a_optLinear}V + {b_optLinear}$'
    #plt.plot(x_linear, y_linear, color=colors[i], label = 'best fit curve $I={0:.2f}V+{1:.2f}$'.format(*poptLinear))
    #labelExp = f'$I = {a_optExp} + {b_optExp}\exp{}$'
    #plt.plot(x_exp, y_exp, color=colors[i], label = 'best fit curve $I={0:.2f}+{1:.10f} \exp {2:.2f}V$'.format(*poptExp))
    
    #plot combined fit
    plt.plot(x_combined, y_combined, color=colors[i], label = 'best fit curve $I={0:.2f}V+{1:.10f} \exp {2:.2f}V$'.format(*poptCombined))

    plt.xlabel('$V/\mathrm{kV}$')
    plt.ylabel('$\mathrm{I}/\mathrm{\mu A}$')
    plt.ylim(0)
    title = str(files[i])
    plt.title(title)
    voltage, current = critVoltage(linear, df, poptLinear, poptExp)
    voltageStr = str(round(voltage, 2))
    plt.scatter(voltage, current, color='r', marker='x', s=100, zorder=10, label = "Critical point = " + voltageStr + '$\mathrm{kV}$')
    voltage1 = voltage1uA(poptExp)
    voltage1str = str(round(voltage1, 2))
    plt.plot(np.ones(10)*voltage1, np.linspace(0,1,10), label = '$V(I = 1\mathrm{\mu A}) = $' + voltage1str +'$\mathrm{kV}$', color='r', linestyle='--')
    plt.plot(np.linspace(0,voltage1,10), np.ones(10), color='r', linestyle='--')
    plt.legend()

    #plt.savefig(title + ' Residuals.png')
    plt.show()

    
    