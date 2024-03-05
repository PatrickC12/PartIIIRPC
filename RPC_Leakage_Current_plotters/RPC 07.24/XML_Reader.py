import numpy as np
import xml.etree.ElementTree as ET
import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re

current_directory = os.getcwd()


#PARSE WITHBOOK DATA
file_path_with_books = current_directory + "\\RPC_Leakage_Current_plotters\\RPC 07.24\\PartIIIRPCwithBooks_modified.xml"
#parser = ET.XMLParser(encoding="unicode_escape")
tree1 = ET.parse(file_path_with_books)
root1 = tree1.getroot()

# Find the ROWDATA element
rowdata1 = root1.find('ROWDATA')

# Create an empty DataFrame to store the attributes
df_with_books = pd.DataFrame(columns=['No', 'time', 'Value', 'Unit'])  # Add column names as needed

#Converting xml data into pandas dataframe.
if rowdata1 is not None:
    # Iterate through each child element of ROWDATA
    for child in rowdata1:
        # Extract attributes from the child element
        no = int(child.get('No'))
        time = child.get('time')
        value = float(child.get('Value'))
        unit = child.get('Unit')

        # Add the attributes to the DataFrame
        df_with_books = df_with_books.append({'No': no, 'time': time, 'Value': value, 'Unit': unit}, ignore_index=True)
else:
    print("ROWDATA element not found in the XML.")

####################################################

#PARSE WITHOUTBOOK DATA
file_path_without_books = current_directory + "\\RPC_Leakage_Current_plotters\\RPC 07.24\\PartIIIRPCwithoutBooks_modified.xml"
#parser = ET.XMLParser(encoding="unicode_escape")
tree2 = ET.parse(file_path_without_books)
root2 = tree2.getroot()

# Find the ROWDATA element
rowdata2 = root2.find('ROWDATA')

# Create an empty DataFrame to store the attributes
df_without_books = pd.DataFrame(columns=['No', 'time', 'Value', 'Unit'])  # Add column names as needed

#Converting xml data into pandas dataframe.
if rowdata2 is not None:
    # Iterate through each child element of ROWDATA
    for child in rowdata2:
        # Extract attributes from the child element
        no = int(child.get('No'))
        time = child.get('time')
        value = float(child.get('Value'))
        unit = child.get('Unit')

        # Add the attributes to the DataFrame
        df_without_books = df_without_books.append({'No': no, 'time': time, 'Value': value, 'Unit': unit}, ignore_index=True)
else:
    print("ROWDATA element not found in the XML.")

#NOW CONVERT THE DATA INTO SOMETHING USEFUL.
#REMOVE OHM READINGS!
df_without_books = df_without_books[(df_without_books['Unit'] == 'mV') | (df_without_books['Unit'] == 'V')]
df_with_books = df_with_books[(df_with_books['Unit'] == 'mV') | (df_with_books['Unit'] == 'V')]

#Convert all volts to same units.
voltage_rows1 = df_without_books[df_without_books['Unit']=='V']
voltage_rows1['Value'] *= 1000
voltage_rows1['Unit'] = 'mV'
df_without_books.update(voltage_rows1)

voltage_rows2 = df_with_books[df_with_books['Unit']=='V']
voltage_rows2['Value'] *= 1000
voltage_rows2['Unit'] = 'mV'
df_with_books.update(voltage_rows2)

#Now measure times relative to initial time (in seconds)
df_with_books['time'] = pd.to_datetime(df_with_books['time'], format='%H:%M:%S').dt.time
df_with_books['time'] = pd.to_timedelta(df_with_books['time'].astype(str))
df_with_books['time'] = (df_with_books['time'] - df_with_books['time'].iloc[0]).dt.total_seconds()

df_without_books['time'] = pd.to_datetime(df_without_books['time'], format='%H:%M:%S').dt.time
df_without_books['time'] = pd.to_timedelta(df_without_books['time'].astype(str))
df_without_books['time'] = (df_without_books['time'] - df_without_books['time'].iloc[0]).dt.total_seconds()

#Now plot residuals.

voltages_with_books = df_with_books['Value'].to_numpy()
times_with_books = df_with_books['time'].to_numpy()

mean_voltage_w_books = 0.35
residuals_w_books = [x-mean_voltage_w_books for x in voltages_with_books]

voltages_without_books = df_without_books['Value'].to_numpy()
times_without_books = df_without_books['time'].to_numpy()

mean_voltage_wo_books = 0.38
residuals_wo_books = [x-mean_voltage_wo_books for x in voltages_without_books]

plt.figure()

plt.plot(times_with_books,residuals_w_books,label='Weight evenly distributed above gas-gap, Voltage constant at 6.43kV')
plt.plot(times_without_books,residuals_wo_books,label='No weight applied to gas-gap from T=0, Voltage = 6.48kV (below breakdown)')

plt.xlabel('Time/seconds')
plt.ylabel('Voltage residual from steady value/ mV')

plt.annotate(f'Mean voltage w/ pressure = {mean_voltage_w_books:.2f}/mV', xy=(0.75, 0.80), xycoords='axes fraction', color='black')
plt.annotate(f'Mean voltage w/o pressure = {mean_voltage_wo_books:.2f}/mV', xy=(0.75,0.75), xycoords='axes fraction', color='black')
plt.title('Voltage fluctuations from mean over period of ~30 minutes')


plt.legend()
plt.show()



