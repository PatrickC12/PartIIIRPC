###################################################################################################################
#The Purpose of this code is to simulate the passage and detection of Muons through a layered RPC Tracking station#
###################################################################################################################

#Generate Muon population.
#Generate random velocity from zenith angle distribution.
#Measure efficiency of single RPC in lab using scintillator and RPC setup.

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

# Setting the Seaborn theme
sns.set_theme(style="darkgrid")

class RPC:

    def __init__(self,gas_mixture,efficiency,dimensions,height,voltage):

        self.gas_mixture = gas_mixture
        #Enter gas mixture as array from GUI

        self.efficiency = efficiency
        #Enter empirically determined efficiency of RPC
        #Clustering ?

        self.dimensions = dimensions
        #Dimensions of RPC 

        self.height = height
        #Z position of this specific RPC.

        self.voltage = voltage
        #Voltage applied across electrodes in the RPC, measuredin kV.

    #RPC will have attributes of dimensions, efficiency, gas mixture etc...
    #Use Garfield++ to find breakdown voltages of gas mixture
    #Experimentally determine which breakdown voltage would be good to use. 

    #Prompt for gas mixture, prompt for voltage for each detector would be good. 
    #Coordinates for RPCs.


    #Use GUI interface, generate stack of RPCs, choose gas mixtures and voltages for each. Run simulation of muon count.
        
class RPCSimulatorApp:

    def __init__(self, master):

        self.master = master
        master.title("RPC Tracking station simulation")

        self.frame = ttk.Frame(master)
        self.frame.pack(padx=150, pady=200)

        # #Input how many RPC Plates you would like.

        # self.Number_RPCs = tk.IntVar()
        # self.Number_RPCs = ttk.Entry(self.frame, textvariable = self.Number_RPCs)
        # self.Number_RPCs.pack(pady=5)
        # self.Number_RPCs_label = ttk.Label(self.frame, text= "How many RPCs would you like to simulate: ")
        # self.Number_RPCs_label.pack(pady=5)

        # Button to start generating RPC list
        self.rpc_list = []

        self.manage_rpc_button = ttk.Button(self.frame, text="Manage RPC Plates", command=self.manage_rpc_window)
        self.manage_rpc_button.pack(pady=5)

        self.calc_button = ttk.Button(self.frame, text="Calculate Efficiencies", state='disabled', command=self.calc_efficiencies)
        self.calc_button.pack(pady=5)

        self.log_button = ttk.Button(self.frame, text="Save/Load RPC Log", command=self.log_rpc_window)
        self.log_button.pack(pady=5)
        
        self.simulate_button = ttk.Button(self.frame, text="Run Simulation", command=self.run_simulation_window)
        self.simulate_button.pack(pady=5) 
        #Calculate the track reconstruction efficiencies.
        #Hit reconstruction efficiencies. 
    
###################################################################################################################
#RPC Management Section
################################################################################################################### 
    def manage_rpc_window(self):
        manage_window = tk.Toplevel(self.master)
        manage_window.title("Manage RPC Plates")
        
        self.frame = ttk.Frame(manage_window)
        self.frame.pack(padx=150, pady=200)

        self.add_rpc_button = ttk.Button(manage_window, text="Add RPC Plate", command=self.create_rpc_window)
        self.add_rpc_button.pack(pady=5)

        self.rpc_combobox = ttk.Combobox(manage_window, state="readonly")
        self.rpc_combobox.pack(padx=80)
        self.update_rpc_combobox()

        self.remove_rpc_button = ttk.Button(manage_window, text="Remove RPC Plate", command=self.remove_rpc)
        self.remove_rpc_button.pack(pady=5)
        
        self.plot_button = ttk.Button(manage_window, text="Plot the RPC Setup", command=self.plot_stations_3d)
        self.plot_button.pack(pady=5)
    
    def create_rpc_window(self):
        rpc_window = tk.Toplevel(self.master)
        rpc_window.title("Add RPC Plate")

        # Generate UI elements for RPC attributes
        self.create_rpc_attributes_ui(rpc_window)

        save_button = ttk.Button(rpc_window, text="Save RPC", command=lambda: self.save_rpc(rpc_window))
        save_button.pack(pady=5)
    
    def remove_rpc(self):
        selection_index = self.rpc_combobox.current()
        if selection_index >= 0:  # A selection is made
            self.rpc_list.pop(selection_index)
            self.update_rpc_combobox()  # Update combobox after removal
#Could have done the create_rpc_attributes_ui Better tbh
    def show_entry(self, var, widget):
        #Decide whether or not to enable a widget passed on a user checkbox
        if var.get():
            widget.configure(state='normal')
        else:
            widget.configure(state='disabled')

    def update_rpc_list(self):
        rpc_descriptions = [f"RPC {idx+1}: Height={rpc.height}m, Dimensions={rpc.dimensions}m" for idx, rpc in enumerate(self.rpc_list)]
        self.rpc_combobox['values'] = rpc_descriptions
        if rpc_descriptions:
            self.rpc_combobox.current(0)  # Select the first item by default
        else:
            self.rpc_combobox.set('')

    def update_rpc_combobox(self):
        rpc_descriptions = [f"RPC {idx+1}: Height={rpc.height}m, Dimensions={rpc.dimensions}m" for idx, rpc in enumerate(self.rpc_list)]
        self.rpc_combobox['values'] = rpc_descriptions
        if rpc_descriptions:
            self.rpc_combobox.current(0)
        else:
            self.rpc_combobox.set('')
    
    def create_rpc_attributes_ui(self, rpc_window):


        #UI Elements for entering the attributes of the RPC being added.

        # Height of RPC
        self.height_var_label = ttk.Label(rpc_window, text="Height (in metres) of the RPC plate: ")
        self.height_var_label.pack(pady=5)
        self.height_var = tk.DoubleVar()
        self.height_var_entry = ttk.Entry(rpc_window, textvariable=self.height_var)
        self.height_var_entry.pack(pady=5)
        
        # Voltage across the RPC plate in kV
        self.voltage_var_label = ttk.Label(rpc_window, text="Voltage applied across the RPC electrode (kV): ")
        self.voltage_var_label.pack(pady=5)
        self.voltage_var = tk.DoubleVar()
        self.voltage_var_entry = ttk.Entry(rpc_window, textvariable=self.voltage_var)
        self.voltage_var_entry.pack(pady=5)
        
        # Dimensions of RPC (assumed rectangular)
        self.x_var_label = ttk.Label(rpc_window, text="Width of RPC (m): ")
        self.x_var_label.pack(pady=5)
        self.x_var = tk.DoubleVar()
        self.x_var_entry = ttk.Entry(rpc_window, textvariable=self.x_var)
        self.x_var_entry.pack(pady=5)
        
        self.y_var_label = ttk.Label(rpc_window, text="Length of RPC (m): ")
        self.y_var_label.pack(pady=5)
        self.y_var = tk.DoubleVar()
        self.y_var_entry = ttk.Entry(rpc_window, textvariable=self.y_var)
        self.y_var_entry.pack(pady=5)

        self.t_var_label = ttk.Label(rpc_window, text="Thickness of RPC (mm): ")
        self.t_var_label.pack(pady=5)
        self.t_var = tk.DoubleVar()
        self.t_var_entry = ttk.Entry(rpc_window, textvariable=self.t_var)
        self.t_var_entry.pack(pady=5)
        
        # Gas mixture of RPC
        self.gases = ["Isobutane", "Argon", "CO2", "N2"]
        self.selected_gases = {gas: (tk.BooleanVar(), tk.DoubleVar()) for gas in self.gases}
        self.gas_percentage = {}
        

        for gas in self.gases:

            # Gas percentage entry box
            gas_frame = ttk.Frame(rpc_window)
            gas_frame.pack(side="top", fill="x", pady=5)

            self.gas_percentage_var = tk.DoubleVar()
            self.gas_percentage_entry = ttk.Entry(gas_frame, textvariable=self.gas_percentage_var, state="disabled")
            self.gas_percentage_entry.pack(side="left", padx=5)

            # Checkbox
            self.select_gas = tk.BooleanVar()
            chk = ttk.Checkbutton(rpc_window, text=gas, variable=self.select_gas, command=lambda v=self.select_gas, e=self.gas_percentage_entry: self.show_entry(v, e))
            chk.pack(side="top", anchor="w", pady=5)

            # Gas percentage label
            self.gas_percentage_var_label = ttk.Label(gas_frame, text="% Of Gas mixture by volume: ")
            self.gas_percentage_var_label.pack(side="left")

            self.gas_percentage[gas]=self.gas_percentage_var.get()
                    
        # Efficiency of RPC
        self.efficiency_var_label = ttk.Label(rpc_window, text="Hit efficiency of the RPC: ")
        self.efficiency_var_label.pack(pady=5)
        self.efficiency_var = tk.DoubleVar()
        self.efficiency_var_entry = ttk.Entry(rpc_window, textvariable=self.efficiency_var)
        self.efficiency_var_entry.pack(pady=5)
    
###################################################################################################################
#Data Logging section
################################################################################################################### 
#save_rpc function saves the rpc configuration temporarily in combobox
    def save_rpc(self, rpc_window):

        # Get user inputs and create RPC object

        height = float(self.height_var.get())
        voltage = float(self.voltage_var.get())
        dimensions = [float(self.x_var.get()), float(self.y_var.get()),float(self.y_var.get())]
        efficiency = float(self.efficiency_var.get())
        gas_mixture = {gas: percentage.get() for gas, (selected, percentage) in self.selected_gases.items() if selected.get()}
        new_rpc = RPC(height=height,efficiency=efficiency,
                        dimensions=dimensions,voltage=voltage,gas_mixture=gas_mixture)
        
        # Add RPC object to the array
        self.rpc_list.append(new_rpc)
        
        self.update_rpc_combobox()

        # Close the RPC window
        rpc_window.destroy()

    def log_rpc_window(self):
        log_window = tk.Toplevel(self.master)
        log_window.title("RPC Log Operations")

        self.save_log_button = ttk.Button(log_window, text="Save RPC Setup", command=self.save_rpc_log)
        self.save_log_button.pack(pady=5)

        self.load_log_button = ttk.Button(log_window, text="Load RPC Setup", command=self.load_rpc_log)
        self.load_log_button.pack(pady=5)
#This function saves the data permanently
    def save_rpc_log(self):
        filepath = filedialog.asksaveasfilename(defaultextension="txt", filetypes=[("Text Files", "*.txt")])
        if filepath:
            with open(filepath, "w") as log_file:
                for rpc in self.rpc_list:
                    log_entry = f"{rpc.height},{rpc.voltage},{rpc.dimensions[0]},{rpc.dimensions[1]},{rpc.dimensions[2]},{rpc.efficiency},{rpc.gas_mixture}\n"
                    log_file.write(log_entry)
            messagebox.showinfo("Success", "RPC setup saved successfully.")

    def load_rpc_log(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            with open(filepath, "r") as log_file:
                self.rpc_list.clear()
                for line in log_file:
                    height, voltage, width, length, thickness, efficiency, gas_mixture = line.strip().split(',')
                    rpc = RPC(height=float(height), efficiency=float(efficiency), dimensions=[float(width), float(length), float(thickness)], voltage=float(voltage), gas_mixture=eval(gas_mixture))
                    self.rpc_list.append(rpc)
###################################################################################################################
#3D plot section
################################################################################################################### 
    def plot_stations_3d(self, df=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for rpc in self.rpc_list:
            z = rpc.height
            width, length, _ = rpc.dimensions

            vertices = np.array([[0, 0, z],
                                [width, 0, z],
                                [width, length, z],
                                [0, length, z]])
            
            # Define the vertices of the rectangle
            faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
            poly3d = Poly3DCollection(faces, alpha=0.5, edgecolors='r', linewidths=1, facecolors='cyan')
            ax.add_collection3d(poly3d)
        
        if df is not None:
            ax.scatter(df['x_position'], df['y_position'], df['z_position_at_detection'], color='red', marker='o', label='Detected Muons')
            ax.legend()

        # Setting the labels for each axis
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Height (m)')
        
        # Auto-scaling the axes to fit all RPC plates
        ax.auto_scale_xyz([0, max(rpc.dimensions[0] for rpc in self.rpc_list)], 
                        [0, max(rpc.dimensions[1] for rpc in self.rpc_list)], 
                        [0, max(rpc.height for rpc in self.rpc_list)])
        
        plt.show()
    
###################################################################################################################
#Simulation Section
################################################################################################################### 
    def run_simulation_window(self):
        simulation_window = tk.Toplevel(self.master)
        simulation_window.title("Simulation Settings")

        # Number of muons
        self.num_muons_label = ttk.Label(simulation_window, text="Number of muons/ns:")
        self.num_muons_label.pack(pady=5)
        self.num_muons_var = tk.IntVar()
        self.num_muons_entry = ttk.Entry(simulation_window, textvariable=self.num_muons_var)
        self.num_muons_entry.pack(pady=5)

        # Simulation time in ns
        self.sim_time_label = ttk.Label(simulation_window, text="Simulation time (ns):")
        self.sim_time_label.pack(pady=5)
        self.sim_time_var = tk.DoubleVar()
        self.sim_time_entry = ttk.Entry(simulation_window, textvariable=self.sim_time_var)
        self.sim_time_entry.pack(pady=5)

        # Start simulation button
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation)
        self.start_sim_button.pack(pady=5)

    def start_simulation(self):
        muons_per_ns = self.num_muons_var.get()
        sim_time = self.sim_time_var.get()
        muon_speed = 0.98
        speed_of_light = 0.299792458 # m/ns
        detected_muons = [] # List to store detected muon data

        for ns in range(int(sim_time)):
            if np.random.uniform(0, 1) < muons_per_ns:
                x_pos = np.random.uniform(0, max(rpc.dimensions[0] for rpc in self.rpc_list))
                y_pos = np.random.uniform(0, max(rpc.dimensions[1] for rpc in self.rpc_list))
                z_pos = max(rpc.height for rpc in self.rpc_list) + 5

                # Check for detection by each RPC plate
                for rpc in self.rpc_list:
                    if rpc.height < z_pos and rpc.height + rpc.dimensions[2]/1000 >= z_pos - muon_speed * speed_of_light * ns:
                        if np.random.uniform(0, 1) <= rpc.efficiency:
                            detection_time = ns
                            detected_muons.append({
                                "x_position": x_pos,
                                "y_position": y_pos,
                                "z_position_at_detection": rpc.height,
                                "detection_time_ns": detection_time,
                                "starting_z_position": z_pos,
                                "initial_velocity": muon_speed * speed_of_light
                            })

        df_detected_muons = pd.DataFrame(detected_muons)

        self.simulation_finished_dialog(df_detected_muons)
###################################################################################################################
#Simulation result section
###################################################################################################################
    def simulation_finished_dialog(self, df_detected_muons):
        dialog_window = tk.Toplevel(self.master)
        dialog_window.title("Simulation Finished")

        # Button to view data in DataFrame
        view_data_button = ttk.Button(dialog_window, text="View Data", command=lambda: self.view_data(df_detected_muons))
        view_data_button.pack(pady=5)

        # Button to plot data on 3D plot
        plot_data_button = ttk.Button(dialog_window, text="Plot Data", command=lambda: self.plot_detected_muons(df_detected_muons))
        plot_data_button.pack(pady=5)

        # Button to save data into a CSV (redundant since data is already saved, but added for completeness)
        save_data_button = ttk.Button(dialog_window, text="Save Data Again", command=lambda: self.save_data_again(df_detected_muons))
        save_data_button.pack(pady=5)
        
        play_video_button = ttk.Button(dialog_window, text="Play Video", command=lambda: self.play_video(df_detected_muons))
        play_video_button.pack(pady=5)
        
    def view_data(self, df):
        if df.empty:
            messagebox.showinfo("No Data", "No muons were detected during the simulation.")
            return

        try:
            from pandastable import Table
            data_window = tk.Toplevel(self.master)
            data_window.title("Detected Muons Data")
            pt = Table(data_window, dataframe=df)
            pt.show()
        except ImportError:
            messagebox.showerror("Import Error", "pandastable module is not installed. Please install it to view data.")

    def plot_detected_muons(self, df):
        self.plot_stations_3d(df)

    def save_data_again(self, df):
        filepath = filedialog.asksaveasfilename(defaultextension="csv", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            df.to_csv(filepath, index=False)
            messagebox.showinfo("Data Saved", "The muons data has been saved to " + filepath)
            
    def play_video(self, df):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize empty arrays to store accumulated positions
        x_accumulated, y_accumulated, z_accumulated = [], [], []

        # Set up plot limits based on the data
        ax.set_xlim(df['x_position'].min(), df['x_position'].max())
        ax.set_ylim(df['y_position'].min(), df['y_position'].max())
        ax.set_zlim(df['z_position_at_detection'].min(), df['z_position_at_detection'].max())

        scat = ax.scatter(x_accumulated, y_accumulated, z_accumulated)

        def animate(i):
            # Get data for the current frame
            current_data = df[df['detection_time_ns'] == i]
            x_current = current_data['x_position'].values
            y_current = current_data['y_position'].values
            z_current = current_data['z_position_at_detection'].values

            # Accumulate the positions
            x_accumulated.extend(x_current)
            y_accumulated.extend(y_current)
            z_accumulated.extend(z_current)

            # Update scatter plot data
            scat._offsets3d = (x_accumulated, y_accumulated, z_accumulated)
            return scat,

        ani = FuncAnimation(fig, animate, frames=len(df['detection_time_ns'].unique()), interval=100)

        plt.show()
        
    def calc_efficiencies(self):
        pass



if __name__ == "__main__":
    with open("rpc_log.txt", 'w') as log_file:
        log_file.write('')
    root = tk.Tk()
    app = RPCSimulatorApp(root)
    root.mainloop()

#Later ideas:
        # Generate a decaying particle, some set lifetime.
        # Create charge products, trace paths of products, do animation.
        # Run example for ANUBIS tracking station. 




