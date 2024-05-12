import warnings
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.ATLAS)

compNo = 1
channels = 4

loc="C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/Scintillator_Plotter/Efficiency_1004"
loc1="C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/Scintillator_Plotter/Comparator tests"

def sort_files(loc):
    fp=[]

    files = glob.glob(os.path.join(loc, "*"))
    files.sort()

    for i in range(compNo):
        fp.append(files[i])
    #print(fp)

    chNames=[]

    for i in range(compNo):
        redunanter, redundant, used = fp[i].partition('C:/Users/tomad/OneDrive - University of Cambridge/Cambridge/Fourth Year/Project/Repo/PartIIIRPC-1/Scintillator_Plotter/Efficiency_1004\\')
        chNames.append(used)

    #print(chNames)

    return fp, chNames

def extract_columns(file_path):
    df = pd.read_csv(file_path, parse_dates=[0])

    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    if df.shape[1] < 3:
        raise ValueError("CSV file must have at least 3 columns")

    time_array = df.iloc[:, 1].to_numpy() * 0.001
    results = {}

    for i in range(2, df.shape[1]):
        count_array = pd.to_numeric(df.iloc[:, i], errors='coerce').to_numpy()
        count_rate = (count_array[-1] - count_array[0]) / (time_array[-1] - time_array[0])
        count_diff = count_array[-1] - count_array[0]
        time_diff = time_array[-1] - time_array[0]

        results[f'count_rate_{i-2}'] = count_rate
        results[f'count_{i-2}'] = count_diff
        results[f'time_diff'] = time_diff

    results['Efficiency_OR'] = results['count_rate_2']/results['count_rate_0']*100
    results['Efficiency_AND'] = results['count_rate_4']/results['count_rate_3']*100

    results['Error_OR'] = np.sqrt((results['Efficiency_OR']/100*(1-results['Efficiency_OR']/100)/results['count_0']))*100
    results['Error_AND'] = np.sqrt((results['Efficiency_AND']/100*(1-results['Efficiency_AND']/100)/results['count_3']))*100

    return results

def analyze_folder(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    #print(files)
    data = []

    for file_path in files:

        voltage_level = int(os.path.basename(file_path).split('.')[0])
        try:
            results = extract_columns(file_path)
            results['voltage'] = voltage_level
            data.append(results)
        except ValueError as e:
            print(f"Skipping {file_path}: {e}")


    df = pd.DataFrame(data)
    return df

def plot_counts(df_results, colors, markers):
    labels = ['SiPM 1', 'SiPM 2', 'Both SiPMs', 'Either SiPM']
    for i in range(compNo):
        for j in range(channels):
            plt.plot(df_results[i]['voltage'], df_results[i][f'count_rate_{j}'], marker =markers[j], markersize=3, linestyle='-', label=labels[j], color=colors[j])

        plt.xlabel('SiPM Threshold Voltage [mV]')
        plt.ylabel('Count Rate [Hz]')
        plt.title('Count Rate vs. Voltage')
        plt.yscale('log')

        plt.legend()
        #plt.savefig('Comparator Alignment 0504 No Adjustment')

        plt.show()

def plot_efficiency(df_results, colors, markers):
    labels=['Scintillator 1', 'Scintillator 2', 'Scintillator 3']
    ranges=[3,4,8]
    i = compNo - 1
    plt.plot(df_results[i]['voltage'], df_results[i]['Efficiency_OR'], color=colors[4], label='OR', marker=markers[i], markersize=3)
    plt.plot(df_results[i]['voltage'], df_results[i]['Efficiency_AND'], color=colors[10], label='AND', marker='D', markersize=3)
    plt.errorbar(df_results[i]['voltage'], df_results[i]['Efficiency_OR'], yerr=df_results[i]['Error_OR'], capsize=5, label='_nolegend_', color = colors[4], markersize=2)
    plt.errorbar(df_results[i]['voltage'], df_results[i]['Efficiency_AND'], yerr=df_results[i]['Error_AND'], capsize=5, label='_nolegend_', color = colors[10], markersize=2)
    plt.plot([720]*11, range(0,110,10), marker='', linestyle='--', label='720mV', color='black')
    plt.xlabel('SiPM threshold voltage [mV]')
    plt.ylabel('Efficiency [%]')
    plt.ylim((0,100))
    plt.title('Scintillator 1 efficiency')

    plt.legend()

    plt.show()


df_results = []

fp, chNames = sort_files(loc)

for i in range(compNo):
  df_results.append(analyze_folder(fp[i]))
  df_results[i]['voltage'] = pd.to_numeric(df_results[i]['voltage'], errors='coerce')
  df_results[i] = df_results[i].sort_values(by=['voltage'], ascending=True)
  #print(df_results[i])

colors = ['#b36305', '#e32017', '#ffd300', '#00782a', '#6950a1', '#f3a9bb', '#a0a5a9','#9b0056','#000000','#003688','#0098d4','#95cdba','#00a4a7','#ee7c0e','#94b817','#e21836' ]
markers = ['o', 'd', 'D', 'd', 'x', 'o']

#plot_counts(df_results, colors, markers)

plot_efficiency(df_results, colors, markers)

