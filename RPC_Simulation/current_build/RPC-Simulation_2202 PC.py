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

    def __init__(self,gas_mixture,efficiency,dimensions, strips, height,voltage):

        self.gas_mixture = gas_mixture
        #Enter gas mixture as array from GUI

        self.efficiency = efficiency
        #Enter empirically determined efficiency of RPC
        #Clustering ?

        self.dimensions = dimensions
        #Dimensions of RPC in metres

        self.height = height
        #Z position of this specific RPC.

        self.voltage = voltage
        #Voltage applied across electrodes in the RPC, measured in kV.
        
        self.strips = strips

    #RPC will have attributes of dimensions, efficiency, gas mixture etc...
    #Use Garfield++ to find breakdown voltages of gas mixture
    #Experimentally determine which breakdown voltage would be good to use. 

    #Prompt for gas mixture, prompt for voltage for each detector would be good. 
    #Coordinates for RPCs.


    #Use GUI interface, generate stack of RPCs, choose gas mixtures and voltages for each. Run simulation of muon count.
        
class muon:

    def __init__(self,position,velocity):

        #[x,y,z] coordinates of muon
        self.position = position
        #List of [v_x,v_y,v_z] velocity components of muon. In units of c.
        self.velocity = velocity

        #History of positions and time step

        self.trajectory = []

    def update_position(self,time_step):

        #Update the muon's current position due to its velocity.
        #Muons assumed to MIPs, such that their velocity is roughly constant over the simulation.

        #time_step is in units of nano-seconds ns.

        speed_of_light = 0.299792458 # m/ns

        self.position+= np.multiply(self.velocity,speed_of_light*time_step)

    def simulate_path(self,rpc_list, initial_time,time_step):
        #Simulate path of muon, given time_step etc...

        #Append initial position
        self.trajectory.append([self.position,initial_time])

        #Running time counter
        T = initial_time
        #time step for trajectory simulation. This will be smaller than the time step between generation events.

        dT = time_step

        #Stop simulating the muon's trajectory once it's z cooridnate passes 0.5 m below the lowest RPC.
        while self.position[2] > min(rpc.height for rpc in rpc_list):

            T+=dT
            self.update_position(time_step)
            self.trajectory.append([self.position.copy(),T])

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
        self.rpc_combobox.pack(padx=80, pady=100)
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
        self.height_var_label = ttk.Label(rpc_window, text="Z axis (in metres) of the RPC plate: ")
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
        
        self.xs_var_label = ttk.Label(rpc_window, text="Number of strips in x direction: ")
        self.xs_var_label.pack(pady=5)
        self.xs_var = tk.DoubleVar()
        self.xs_var_entry = ttk.Entry(rpc_window, textvariable=self.xs_var)
        self.xs_var_entry.pack(pady=5)
        
        self.ys_var_label = ttk.Label(rpc_window, text="Number of strips in y direction: ")
        self.ys_var_label.pack(pady=5)
        self.ys_var = tk.DoubleVar()
        self.ys_var_entry = ttk.Entry(rpc_window, textvariable=self.ys_var)
        self.ys_var_entry.pack(pady=5)
        
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
        #dimensions in metres.
        dimensions = [float(self.x_var.get()), float(self.y_var.get()),float(self.t_var.get())/1000]
        strips = [int(self.xs_var.get()), int(self.ys_var.get())]
        efficiency = float(self.efficiency_var.get())
        gas_mixture = {gas: percentage.get() for gas, (selected, percentage) in self.selected_gases.items() if selected.get()}
        new_rpc = RPC(height=height, efficiency=efficiency,
                        dimensions=dimensions,strips = strips, voltage=voltage, gas_mixture=gas_mixture)
        
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
                    log_entry = f"{rpc.height},{rpc.voltage},{rpc.dimensions[0]},{rpc.dimensions[1]},{rpc.dimensions[2]},{rpc.strips[0]},{rpc.strips[1]},{rpc.efficiency},{rpc.gas_mixture}\n"
                    log_file.write(log_entry)
            messagebox.showinfo("Success", "RPC setup saved successfully.")

    def load_rpc_log(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            with open(filepath, "r") as log_file:
                self.rpc_list.clear()
                for line in log_file:
                    height, voltage, width, length, thickness, xstrip, ystrip, efficiency, gas_mixture = line.strip().split(',')
                    rpc = RPC(height=float(height), efficiency=float(efficiency), dimensions=[float(width), float(length), float(thickness)],strips=[int(xstrip), int(ystrip)], voltage=float(voltage), gas_mixture=eval(gas_mixture))
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
        self.muon_flux_label = ttk.Label(simulation_window, text="Flux of muons /cm^2/s: ")
        self.muon_flux_label.pack(pady=5)
        self.muon_flux_var = tk.DoubleVar()
        self.muon_flux_entry = ttk.Entry(simulation_window, textvariable=self.muon_flux_var)
        self.muon_flux_entry.pack(pady=5)

        # Simulation time in ns
        self.sim_time_label = ttk.Label(simulation_window, text="Simulation time (s):")
        self.sim_time_label.pack(pady=5)
        self.sim_time_var = tk.DoubleVar()
        self.sim_time_entry = ttk.Entry(simulation_window, textvariable=self.sim_time_var)
        self.sim_time_entry.pack(pady=5)
        
        # Advanced settings button
        self.adv_settings_button = ttk.Button(simulation_window, text="Advanced Settings", command=self.open_advanced_settings)
        self.adv_settings_button.pack(pady=5)

        # Start simulation button
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation)
        self.start_sim_button.pack(pady=5)
        
    def open_advanced_settings(self):
        advanced_window = tk.Toplevel(self.master)
        advanced_window.title("Advanced Settings")

        # Checkbox for using strips
        self.use_strips_var = tk.BooleanVar()
        self.use_strips_check = ttk.Checkbutton(advanced_window, text="Use strips", variable=self.use_strips_var, command=self.toggle_strips)
        self.use_strips_check.pack(pady=5)

        #Add checkbox for enabling dark counts

    def toggle_strips(self):
        togglestrip = self.use_strips_var.get()
               
    def start_simulation(self):

        #Muon flux, muon_flux_var is measured in /cm^2/s
        muons_flux = self.muon_flux_var.get()
        #Now calculate the expected muon rate given the dimensions of the problem.
        area_m2 = max(rpc.dimensions[0] for rpc in self.rpc_list)*max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1025
        #Now calculate the average rate of muon arrival given the problem specifics.    
        rate = muons_flux*area_m2*(1e4)

        #Number of steps per second (eg 1e9 corresponds to timestep of 1ns)
        steps_per_sec = 1e9

        #sim_time in seconds
        sim_time = self.sim_time_var.get()
        #sim time in nanoseconds
        sim_time_ns = sim_time*steps_per_sec

        #muon speed in units of c
        muon_speed = 0.98
        speed_of_light = 0.299792458 # m/ns

        #Create empty area of muons to populate:

        muons = []
    
        #Generate Muons at rate given by poisson distribution.
        #Each time step generate certain number of muons.

        #Trajectory simulator time step, adaptive here since timescale of muon motion much faster than their generation.

        traj_time_step = min(rpc.dimensions[2] for rpc in self.rpc_list)/(muon_speed*speed_of_light)

        for ns in range(int(sim_time_ns)):

            if ns%(1e6) == 0:
                t = ns/(1e9)
                print(f"Time elapse = {t} seconds")
            
            ##Generate new muons in this time step###
            # Num_muons_generated is number of new muons, this is drawn from a Poissonian distribution

            #Problem, could pass through an RPC in a single time step and hence be missed...
            #Ideas, generate muon at timestep ns. 
            #Extrapolate the muons behaviour until it reaches some z coordinate below the lowest RPC.

            #Scaling muon arrival rate to time step (1ns)

            scaled_rate = rate/(steps_per_sec)

            #Generate a certain number of muons in this time step, drawn from a Poisson distribution.

            num_muons_generated = np.random.poisson(scaled_rate)

            if num_muons_generated == 0:
                #If no muons produced in this time step, skip the time step to the next one.
                continue
            else:
                #For each muon generated in this time step:
                #Simulate its trajectory until it passes under a certain z value (eg 1m below the height of the lowest RPC plate)
                for i in range(num_muons_generated):

                    #Generate initial position of muon above the RPC plate

                    x_pos = np.random.uniform(0, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.05)
                    y_pos = np.random.uniform(0, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.05)
                    z_pos = max(rpc.height for rpc in self.rpc_list)+0.5

                    position = [x_pos,y_pos,z_pos]

                    #Generate velocity of muon, cos^2(theta) distribution for angle.

                    phi = np.random.uniform(0.2*np.pi)
                    theta = 0 
                    #Change this later to match angular distribution of cosmic muons.

                    #Create velocity of muon, it is very important to put a - sign on the muon's velocity, or else 

                    velocity = np.multiply(0.98,[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),-np.cos(theta)])
                    
                    #Create object of class muon with these randomly generated velocities and positions.
                    generated_muon = muon(position=position, velocity= velocity)

                    generated_muon.simulate_path(self.rpc_list,initial_time=ns,time_step=traj_time_step)

                    muons.append(generated_muon)

        #Now pass on the generated list of muon trajectories to the simulation result section

        self.simulation_finished_dialog(muons)
###################################################################################################################
#Simulation result section
###################################################################################################################
    def simulation_finished_dialog(self, muons):
        dialog_window = tk.Toplevel(self.master)
        dialog_window.title("Simulation Finished")

        # Button to view data in DataFrame
        view_data_button = ttk.Button(dialog_window, text="View Data", command=lambda: self.view_data(muons))
        view_data_button.pack(pady=5)

        # Button to plot data on 3D plot
        plot_data_button = ttk.Button(dialog_window, text="Plot Data", command=lambda: self.plot_detected_muons(muons))
        plot_data_button.pack(pady=5)

        # Button to save data into a CSV (redundant since data is already saved, but added for completeness)
        save_data_button = ttk.Button(dialog_window, text="Save Data Again", command=lambda: self.save_data_again(muons))
        save_data_button.pack(pady=5)
        
        play_video_button = ttk.Button(dialog_window, text="Play Video", command=lambda: self.play_video(muons))
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
            
    def play_video(self,muons):

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Simulation time in seconds
        sim_time = self.sim_time_var.get()
        
        #desired number of frames, I would like 1 frame for every 1ms of the animation.

        number_of_frames = sim_time *1000

        # Function to update the plot for each frame of the animation
        def update(frame):
            
            # Calculate the time corresponding to the current frame, I have set it so that 1 frame is 1 millisecond.

            time = frame/1000

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.05)
            ax.set_ylim(0, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.05)
            ax.set_zlim(0, max(rpc.height for rpc in self.rpc_list) + 0.5)

            for muon in muons:
                
                if muon.trajectory[0][1] <= time:
                        position = np.array(muon.trajectory)[:, 0]  # Extract positions from the trajectory
                        x, y, z = position[:, 0], position[:, 1], position[:, 2]
                        ax.plot(x[:frame], y[:frame], z[:frame], color='red')  # Plot the trajectory
                else:
                    continue

            return ax

        # Create the animation
        ani = FuncAnimation(fig, update, frames=number_of_frames, interval=50)

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




