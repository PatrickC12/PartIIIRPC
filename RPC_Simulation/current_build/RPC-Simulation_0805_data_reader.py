#Read the data from RPC sim

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

#hep.style.use(hep.style.ATLAS)

data_path = "C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/RPC_Simulation/pc_demos/17strip/1.csv"

df = pd.read_csv(data_path)


df1 = df.groupby('muon_index').filter(lambda x: len(x) == 3)
    

N = len(df1)

N = N - N % 3

num=0
denom=0

for index, group in df1.groupby('muon_index'):
    if group.iloc[0]['Outcome']=='Y':
        if group.iloc[2]['Outcome']=='Y':
            denom += 1
            if group.iloc[1]['Outcome']=='Y':
                num+=1

print(num)
print(denom)
print(num/denom*100)

