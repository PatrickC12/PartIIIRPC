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

#bin ['#ffd300', '#f3a9bb', '#a0a5a9']

colors = ['#b36305', '#e32017',  '#00782a', '#6950a1','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]
markers = ['^', 'v', 'D', 'd', 'x', 'o']
captions=['Freon with increasing isobutane', 'Mix 1 with increasing $\mathrm{CO_2}$', 'CERN mix with increasing $\mathrm{CO_2}$', 'Adding $\mathrm{SF_6}$', 'Isobutane with increasing $\mathrm{CO_2}$']
            

def efficiency_critV():
    data_path = "C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/RPC_Efficiencies/Critical_voltages.csv"
    df = pd.read_csv(data_path)

    labels=['101', '12', '14', '16', '8']
    files=[]
    
    

    for i in range(len(labels)):
        data = df[df['File'].str.partition('g')[0] == labels[i]]
        plt.plot(data['Critical voltage/kV'], data['Max_eff/%'], label=captions[i], linestyle='-', marker=markers[i], markersize=7, color=colors[i])
        plt.errorbar(data['Critical voltage/kV'], data['Max_eff/%'], yerr=data['error_in_max/%'], linestyle='None', capsize=5, label='_nolegend_', color = colors[i])

    plt.xlabel('$V_{\mathrm{crit}}$ [$\mathrm{kV}$]')
    plt.ylabel('Maximum efficiency [%]')
    #plt.ylim(0)
    #title = str(selected_files[i])
    plt.title("Maximum efficiency vs. $V_{\mathrm{crit}}$")
    plt.legend()
    #plt.savefig(title + '.Efficiency.png')
    plt.show()  

    return 

efficiency_critV()

