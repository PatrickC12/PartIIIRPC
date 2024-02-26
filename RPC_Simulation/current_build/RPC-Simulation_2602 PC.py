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
from PIL import ImageTk, Image
# import requests
from io import BytesIO

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

    def coincidence(self):
    
        pass

    def generate_dark(self):

        #incorporate self.strips here somewhere.
        pass

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
        self.times = []

        #Add starting frame number for muon trajectory, useful for some plots

        self.starting_frame = 0

        # List containing information on coordinates and times a muon passes through an RPC plate. Y/N is whether or not the RPC registers the hit.
        # [[x,y,z,t,Y/N],...,[]]

        self.hits = []

    def update_position(self,time_step):

        #Update the muon's current position due to its velocity.
        #Muons assumed to MIPs, such that their velocity is roughly constant over the simulation.

        #time_step is in units of nano-seconds ns.

        speed_of_light = 0.299792458 # m/ns

        self.position+= np.multiply(self.velocity,speed_of_light*time_step)

    def check_hit(self,rpc_list):

        #Get efficiency of RPC from rpc_list, use this to decide if a hit is detected or not.
        pass
    
    def simulate_path(self,rpc_list, initial_time,time_step):
        #Simulate path of muon, given time_step and initial_time in nanoseconds

        #Append initial position
        self.trajectory.append(np.array(self.position))

        #Running time counter, nanoseconds
        T = initial_time
        self.times.append(T)

        #time step for trajectory simulation. This will be smaller than the time step between generation events.

        dT = time_step

        min_rpc_height = min(rpc.height for rpc in rpc_list)
        max_x_dimension = max(rpc.dimensions[0] for rpc in rpc_list)
        max_y_dimension = max(rpc.dimensions[1] for rpc in rpc_list)

        #Stop simulating the muon's trajectory once it's z cooridnate passes 0.5 m below the lowest RPC.

        while (self.position[2] > min_rpc_height and
            -max_x_dimension * 0.1 < self.position[0] < max_x_dimension * 1.1 and
            -max_y_dimension * 0.1 < self.position[1] < max_y_dimension * 1.1):

            T+=dT
            self.update_position(time_step)
            self.times.append(T)
            self.trajectory.append(self.position.copy())

class RPCSimulatorApp:

    def __init__(self, master):

        current_directory = os.getcwd()

        self.master = master
        master.title("Ultimate RPC Tracking station simulation")

        self.style = ttk.Style()
        self.style.configure('custom.TFrame',relief='solid')

        self.frame = ttk.Frame(master,style='custom.TFrame')
        self.frame.pack()

        self.img = Image.open(current_directory + "\RPC_Simulation\images\Banner.png")
        self.img = self.img.resize((400,120))
        self.img_tk = ImageTk.PhotoImage(self.img)

        self.banner = ttk.Label(self.frame, image= self.img_tk)
        self.banner.pack(side='top',pady= 20)

        self.img2 = Image.open(current_directory + "\RPC_Simulation\images\wip.jpg")
        self.img2 = self.img2.resize((300,120))
        self.img_tk2 = ImageTk.PhotoImage(self.img2)

        self.banner = ttk.Label(self.frame, image= self.img_tk2)
        self.banner.pack(side='bottom',pady= 20)

        # Button to start generating RPC list
        self.rpc_list = []

        self.manage_rpc_button = ttk.Button(self.frame, text="Manage RPC Plates (WIP)", command=self.manage_rpc_window)
        self.manage_rpc_button.pack(pady=5)

        self.log_button = ttk.Button(self.frame, text="Save/Load RPC Log", command=self.log_rpc_window)
        self.log_button.pack(pady=20)

        self.nanoscale_sim_desc = tk.Label(self.frame, text='Simulate cosmic ray muons through setup',font = 30)
        self.nanoscale_sim_desc.pack(pady=10) 

        self.simulation_var = tk.BooleanVar(value=True)
        self.simulation_nano_checkbox = ttk.Radiobutton(self.frame, text='Nanoseconds scale simulation', variable = self.simulation_var, value = True)
        self.simulation_nano_checkbox.pack(anchor='center')

        self.simulation_norm_checkbox = ttk.Radiobutton(self.frame, text='Seconds scale simulation (WIP)', variable = self.simulation_var, value = False)
        self.simulation_norm_checkbox.pack(anchor= 'center',pady=10)

        self.simulate_button = ttk.Button(self.frame, text="Run cosmic ray muon simulation", style='success.TButton', command=self.run_simulation)
        self.simulate_button.pack(pady=10) 

        self.nanoscale_sim_desc = tk.Label(self.frame, text='Simulate decaying LLNP (WIP)',font = 30)
        self.nanoscale_sim_desc.pack(pady=10) 

    def run_simulation(self):
        
        #Check that an RPC loadout has been selected.

        if len(self.rpc_list) == 0:
            messagebox.showwarning(title="Warning",message="You have not entered an RPC setup")
            return
        else:

            #Choose simulation type depending on the User's input.
            if self.simulation_var.get():
                self.run_simulation_window_nano()
            else:
                self.run_simulation_window_norm()

    
###################################################################################################################
#RPC Management Section
###Mange_RPC panel is quite shite... could do with an upgrade todo ######
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

            gas_percentage_var = tk.DoubleVar()  # Use local variable instead of self.gas_percentage_var
            gas_percentage_entry = ttk.Entry(gas_frame, textvariable=gas_percentage_var, state="disabled")
            gas_percentage_entry.pack(side="left", padx=5)

            # Checkbox
            select_gas = tk.BooleanVar()  # Use local variable instead of self.select_gas
            chk = ttk.Checkbutton(rpc_window, text=gas, variable=select_gas, command=lambda v=select_gas, e=gas_percentage_entry: self.show_entry(v, e))
            chk.pack(side="top", anchor="w", pady=5)

            # Gas percentage label
            gas_percentage_var_label = ttk.Label(gas_frame, text="% Of Gas mixture by volume: ")
            gas_percentage_var_label.pack(side="left")

            # Add gas selection to dictionary when checkbox is clicked
            self.selected_gases[gas] = (select_gas, gas_percentage_var)  # Store variables, not their current values
                    
        # Efficiency of RPC
        self.efficiency_var_label = ttk.Label(rpc_window, text="Hit efficiency of the RPC: ")
        self.efficiency_var_label.pack(pady=5)
        self.efficiency_var = tk.DoubleVar()
        self.efficiency_var_entry = ttk.Entry(rpc_window, textvariable=self.efficiency_var)
        self.efficiency_var_entry.pack(pady=5)

        if float(self.efficiency_var.get())> 1.0 or float(self.efficiency_var.get())<0.0:
            messagebox.showinfo("Please enter a valid efficiency for the RPC plate")
            return
    
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
                i= 1
                for rpc in self.rpc_list:
                    #Probably relabel the number of strips in x and y direction to theta and phi direction 
                    log_entry = f"RPC {i},\n" \
                            f"Height(m) = {rpc.height},\n" \
                            f"Voltage (kV) = {rpc.voltage},\n" \
                            f"Width (m) = {rpc.dimensions[0]},\n" \
                            f"Length(m) = {rpc.dimensions[1]},\n" \
                            f"Thickness (m) = {rpc.dimensions[2]},\n" \
                            f"Number of Strips in x direction = {rpc.strips[0]},\n" \
                            f"Number of strips in y direction = {rpc.strips[1]},\n" \
                            f"RPC Efficiency = {rpc.efficiency},\n" \
                            f"RPC Gas mixture = {rpc.gas_mixture},\n" \
                            "\n"
                    
                    log_file.write(log_entry)
                    i+=1

            messagebox.showinfo("Success", "RPC setup saved successfully.")

    def load_rpc_log(self):

        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])

        #Read RPC Setup files.
        if filepath:

            with open(filepath, "r") as log_file:

                log_file = list(log_file)

                for index,l in enumerate(log_file):
                    log_file[index]=l[:-2]

                self.rpc_list.clear()
              
                #Store starting line number for specific RPCs
                sent_number = []

                j = 1
                for index,line in enumerate(log_file):
                    if line.replace(" ", "") == f"RPC{j}":
                        sent_number.append(index)
                        j+=1
                    else:
                        continue

                for i in range(len(sent_number)):

                    #For each RPC in the txt file, extract the attributes.
                    if i < len(sent_number)-1:

                        rpc_attributes = log_file[sent_number[i]+1:sent_number[i+1]]

                        filt_item = []

                        for item in rpc_attributes:
                            #Filter out unnecessary aspects of string.
                            item = item[item.find(' = ')+3:]
                            filt_item.append(item)

                        rpc = RPC(height=float(filt_item[0]), voltage=float(filt_item[1]), dimensions=[float(filt_item[2]),
                                float(filt_item[3]), float(filt_item[4])],strips=[int(filt_item[5]), int(filt_item[6])], 
                                efficiency=float(filt_item[7]), gas_mixture=eval(filt_item[8]))
                        self.rpc_list.append(rpc)

                        print(f"RPC {i} successfully added")
                    else:

                        rpc_attributes = log_file[sent_number[i]+1:]

                        filt_item = []

                        for item in rpc_attributes:

                            item = item[item.find(' = ')+3:]
                            filt_item.append(item)

                        rpc = RPC(height=float(filt_item[0]), voltage=float(filt_item[1]), dimensions=[float(filt_item[2]),
                                float(filt_item[3]), float(filt_item[4])],strips=[int(filt_item[5]), int(filt_item[6])], 
                                efficiency=float(filt_item[7]), gas_mixture=eval(filt_item[8]))
                        self.rpc_list.append(rpc)

                        print(f"RPC {i} successfully added")

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
            pass

        # Setting the labels for each axis
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Height (m)')
        
        # Auto-scaling the axes to fit all RPC plates
        ax.auto_scale_xyz([0, max(rpc.dimensions[0] for rpc in self.rpc_list)], 
                        [0, max(rpc.dimensions[1] for rpc in self.rpc_list)], 
                        [min(rpc.height for rpc in self.rpc_list)-1, max(rpc.height for rpc in self.rpc_list)])
        
        plt.show()
    
###################################################################################################################
# Nanosecond Scale Simulation Section
################################################################################################################### 
    def run_simulation_window_nano(self):

        simulation_window = tk.Toplevel(self.master)
        simulation_window.title("Nanosecond Scale Simulation Settings")

        # Number of muons
        self.muon_flux_label = ttk.Label(simulation_window, text="Flux of muons /cm\u00b2/ns: ")
        self.muon_flux_label.pack(pady=5)
        self.muon_flux_var = tk.DoubleVar()
        self.muon_flux_entry = ttk.Entry(simulation_window, textvariable=self.muon_flux_var)
        self.muon_flux_entry.pack(pady=5)

        # Simulation time in ns
        self.sim_time_label = ttk.Label(simulation_window, text="Simulation time (ns):")
        self.sim_time_label.pack(pady=5)
        self.sim_time_var = tk.DoubleVar()
        self.sim_time_entry = ttk.Entry(simulation_window, textvariable=self.sim_time_var)
        self.sim_time_entry.pack(pady=5)
        
        # Advanced settings button
        self.adv_settings_button = ttk.Button(simulation_window, text="Advanced Settings", command=self.open_advanced_settings)
        self.adv_settings_button.pack(pady=5)

        # Start simulation button
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation_nanoscale)
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
        pass
               
    def start_simulation_nanoscale(self):

        ##Nanoscale algorithm uses ineffecient timestep method. Here I simulate all steps regardless of whether a muon is generated or not.
        ##I would like to update this algorithm at some point, rather draw the time t for 1 event to occur from a poisson distribution and jump to that
        #step.

        #Muon flux, muon_flux_var is measured in /cm^2/ns
        muons_flux = self.muon_flux_var.get()
        #Now calculate the expected muon rate given the dimensions of the problem.
        area_m2 = max(rpc.dimensions[0] for rpc in self.rpc_list)*max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1025
        #Now calculate the average rate of muon arrival given the problem specifics.    
        rate = muons_flux*area_m2*(1e4)

        #sim_time in nanoseconds
        sim_time_ns = self.sim_time_var.get()

        #Create empty area of muons to populate:

        muons = []
    
        #Generate Muons at rate given by poisson distribution.
        #Each time step generate certain number of muons.

        #Trajectory simulator time step (in nanosceonds).
        #Timescale always shorter than the vertical passage time of straight down muon through RPC plates, this hopefully means all muons have
        #chance to be detected by simulation.

        traj_time_step = min(rpc.dimensions[2] for rpc in self.rpc_list) / (0.299792458)

        #Convert continuous probability distribution function into discrete distribution function for zenith angle
        theta_val = np.linspace(0,np.pi/2,100)
        probs = [4/(np.pi) * (np.cos(x))**2 for x in theta_val]
        Norm = np.sum(probs)
        norm_probs = np.multiply(1/Norm,probs)

        for ns in range(int(sim_time_ns)):

            """ if ns%(1e6) == 0:
                t = ns/(1e9)
                print(f"Time elapse = {t} seconds") """
            
            ##Generate new muons in this time step###
            # Num_muons_generated is number of new muons, this is drawn from a Poissonian distribution

            #Problem, could pass through an RPC in a single time step and hence be missed...
            #Ideas, generate muon at timestep ns. 
            #Extrapolate the muons behaviour until it reaches some z coordinate below the lowest RPC.

            #Scaling muon arrival rate to time step (1ns)

            scaled_rate = rate

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
                    z_pos = max(rpc.height for rpc in self.rpc_list)

                    position = [x_pos,y_pos,z_pos]

                    #####Generate velocity of muon, cos^2(theta) distribution for angle.

                    phi = np.random.uniform(0,2*np.pi)

                    #Generating discrete distribution for zenith angle of muon.

                    theta = np.random.choice(theta_val,p=norm_probs)

                    #Create velocity of muon, it is very important to put a - sign on the muon's velocity, or else 

                    velocity = np.multiply(0.98,[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),-np.cos(theta)])
                    
                    #Create object of class muon with these randomly generated velocities and positions.
                    generated_muon = muon(position=position, velocity= velocity)

                    generated_muon.simulate_path(self.rpc_list,initial_time=ns,time_step=traj_time_step)

                    muons.append(generated_muon)

        #Now pass on the generated list of muon trajectories to the simulation result section
                    
        self.simulation_finished_dialog(muons)

###################################################################################################################
# Second Scale Simulation Section
################################################################################################################### 
    def run_simulation_window_norm(self):

        simulation_window = tk.Toplevel(self.master)
        simulation_window.title("Second Scale Simulation Settings")

        # Number of muons
        self.muon_flux_label = ttk.Label(simulation_window, text="Flux of muons /cm\u00b2/s: ")
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
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation_normscale)
        self.start_sim_button.pack(pady=5)
        
    def open_advanced_settings(self):

        advanced_window = tk.Toplevel(self.master)
        advanced_window.title("Advanced Settings")

        # Checkbox for using strips
        self.use_strips_var = tk.BooleanVar()
        self.use_strips_check = ttk.Checkbutton(advanced_window, text="Use strips", variable=self.use_strips_var, command=self.toggle_strips)
        self.use_strips_check.pack(pady=5)

        #Add checkbox for enabling dark counts

        self.use_dark_counts_var = tk.BooleanVar()
        self.use_dark_counts_check = ttk.Checkbutton(advanced_window, text="Use strips", variable=self.use_dark_counts_var, command=self.toggle_strips)
        self.use_dark_counts_check.pack(pady=5)

    def toggle_strips(self):
        togglestrip = self.use_strips_var.get()
        pass
               
    def start_simulation_normscale(self):

        #Simulation time in seconds.
        sim_time = self.sim_time_var.get()

        #Muon flux, muon_flux_var is measured in /cm^2/s
        muons_flux = self.muon_flux_var.get()
        #Now calculate the expected muon rate given the dimensions of the problem.
        area_m2 = max(rpc.dimensions[0] for rpc in self.rpc_list)*max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1025
        #Now calculate the average rate of muon arrival (/sec) given the problem specifics.    
        rate = muons_flux*area_m2*(1e4)

        muons = []
    
        #Start counting time
        running_time = 0

        traj_time_step = min(rpc.dimensions[2] for rpc in self.rpc_list) / (0.299792458)

        #Convert continuous probability distribution function into discrete distribution function for zenith angle
        theta_val = np.linspace(0,np.pi/2,100)
        probs = [4/(np.pi) * (np.cos(x))**2 for x in theta_val]
        Norm = np.sum(probs)
        norm_probs = np.multiply(1/Norm,probs) 

        while running_time < sim_time:

            #Instead of simulating each time step, I am now drawing the time t of the next muon hit from a probability distribution.
            #The distribution used here is a first passage time distribution. For a poisson process this leads to an exponential distribution.
            #Due to the exponential nature of this first passage time distribution, it makes it difficult to sample discretely from it.
            #Have to use inverse transform sampling from the first passage time distribution instead.

            u = np.random.uniform()
            #Luckily np.random.uniform() excludes 1, as this produces an infinite result from inverse transform sampling...

            #muon_time here is the time between when the last muon was produced and the next muon is generated.
            muon_time = -1 / rate *np.log(1-u)

            #running_time (in seconds) now updated to time when new muon is generated
            running_time+= muon_time

            #it is possible for running_time to exceed sim_time in the while loop now, just including this break to avoid this.

            if running_time >= sim_time:
                break

            #Generate initial position of muon above the RPC plate

            x_pos = np.random.uniform(0, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.05)
            y_pos = np.random.uniform(0, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.05)
            z_pos = max(rpc.height for rpc in self.rpc_list)

            position = [x_pos,y_pos,z_pos]

            #####Generate velocity of muon, cos^2(theta) distribution for angle.

            phi = np.random.uniform(0,2*np.pi)
            theta = np.random.choice(theta_val,p=norm_probs)

            #Create velocity of muon, it is very important to put a - sign on the muon's velocity, or else 

            velocity = np.multiply(0.98,[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),-np.cos(theta)])
            
            #Create object of class muon with these randomly generated velocities and positions.
            generated_muon = muon(position=position, velocity= velocity)

            generated_muon.simulate_path(self.rpc_list,initial_time=(running_time*1e9),time_step=traj_time_step)

            muons.append(generated_muon)

        self.simulation_finished_dialog(muons)
###################################################################################################################
###################################################################################################################
#Simulation result section
###################################################################################################################
    def simulation_finished_dialog(self, muons):

        dialog_window = tk.Toplevel(self.master)
        dialog_window.title(f"Simulation Finished")

        dialog_window_desc = tk.Label(dialog_window, text=f'{len(muons)} muons generated')
        dialog_window_desc.pack(padx=30, pady=30)

        # Button to plot data on 3D plot
        plot_data_button = ttk.Button(dialog_window, text="Plot Data", command=lambda: self.plot_detected_muons(muons))
        plot_data_button.pack(pady=5)

        # Button to save data into a CSV
        save_data_button = ttk.Button(dialog_window, text="Save Data Again", command=lambda: self.save_data_again(muons))
        save_data_button.pack(pady=5)
        
        play_video_button = ttk.Button(dialog_window, text="Play Video", command=lambda: self.play_video(muons))
        play_video_button.pack(pady=5)

    def play_video(self,muons):

        #Select video player format if nanoscale or norm scale simulation.
        if self.simulation_var.get():
            self.play_video_nano(muons)
        else:
            self.play_video_norm(muons)

    def plot_detected_muons(self, df):
        self.plot_stations_3d(df)

    def save_data_again(self, muons):
        filepath = filedialog.asksaveasfilename(defaultextension="csv", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            muons.to_csv(filepath, index=False)
            messagebox.showinfo("Data Saved", "The muons data has been saved to " + filepath)
            
    def play_video_nano(self,muons):

        # Create a figure and a 3D axis
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

        #Simulation time in nanoseconds
        sim_time = self.sim_time_var.get()
        
        #desired number of frames, I would like 1 frame for every 1ns of the animation.

        number_of_frames = int(sim_time)

        # Function to update the plot for each frame of the animation
        def update(frame):

            #If you would like to plot the entire trajectory (eg if you have a low flux and only a small # events) then just remove this ax.cla()
            
            ax.cla()

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
            
            # Calculate the time corresponding to the current frame, I have set it so that 1 frame is 1 nanosecond

            """ #Only clear frames after every 5 frames, this increases the length of the streams behind the muons.
            if frame % 5 ==0:
                for line in ax.lines:
                    if len(line.get_xdata()) > 5:  # Keep a maximum of 100 points per trajectory
                        line.remove()

                ####OLD SOLUTION ####

                """ """ ax.cla()

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
                    ax.add_collection3d(poly3d) """ """ """

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-max(rpc.dimensions[0] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.1)
            ax.set_ylim(-max(rpc.dimensions[1] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1)
            ax.set_zlim(0, max(rpc.height for rpc in self.rpc_list) + 2)

            for muon in muons:

                if muon.times[0] <= frame <= muon.times[-1]:

                    comb =np.hstack((muon.trajectory, np.array(muon.times)[:, np.newaxis]))
                    
                    filtered_trajectory = [x for x in comb if frame-1<x[3]<=frame]

                    if len(filtered_trajectory)==0:
                        print(f"EMPTY FILTER!!, Frame is {frame}")
                        pass
                    else:
                        # Extract positions from the trajectory
                        position = np.array(filtered_trajectory)[:, :3]
                        x, y, z = position[:, 0], position[:, 1], position[:, 2]
                        ax.plot(x, y, z, color='red')  # Plot the trajectory

                        #Uncomment to add tail if you would like.

                        """ # Plot past trajectory points tails.
                        if frame > 1:
                            past_frame = frame - 1
                            past_filtered_trajectory = [x for x in comb if past_frame - 1 < x[3] <= past_frame]
                            if len(past_filtered_trajectory) != 0:
                                past_position = np.array(past_filtered_trajectory)[:, :3]
                                x_past, y_past, z_past = past_position[:, 0], past_position[:, 1], past_position[:, 2]
                                ax.plot(x_past, y_past, z_past, color='darkred') """
                            
                else:
                    continue
                

                  # Add text annotation for simulation time
                ax.annotate(f'Simulation time/s = {frame}', xy=(0.05, 0.95), xycoords='axes fraction', color='black')

            return ax

        # Create the animation
        ani = FuncAnimation(fig, update, frames=number_of_frames, interval=50)

        plt.show()

    def play_video_norm(self,muons):

        #Making a copy of muons so that the original data is not deleted...
        muons_c = muons.copy()

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Plot RPC stations
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

        #Simulation time in seconds
        sim_time = self.sim_time_var.get()
        #desired number of frames, I would like 1 frame for every 10 ms of the animation.
        number_of_frames = int(sim_time)*100

        # Function to update the plot for each frame of the animation
        def update(frame):

            #If you would like to plot the entire trajectory (eg if you have a low flux and only a small # events) then just remove this ax.cla()
            
            ax.cla()

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
    
            #1 frame = 10 ms  of passage time in the simulation.
            #time in s
            time = (frame)*(0.01)

            time_scaled = time*(1e9)
    
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-max(rpc.dimensions[0] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.1)
            ax.set_ylim(-max(rpc.dimensions[1] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1)
            ax.set_zlim(0, max(rpc.height for rpc in self.rpc_list) + 2)

            for muon in muons_c:

                if muon.times[0] <= time_scaled:

                    if muon.starting_frame == 0:
                        muon.starting_frame = frame
                    else:
                        pass

                    comb =np.hstack((muon.trajectory, np.array(muon.times)[:, np.newaxis]))

                # Extract positions from the trajectory
                    position = np.array(comb)[:, :3]
                    x, y, z = position[:, 0], position[:, 1], position[:, 2]
                    ax.plot(x, y, z, color='red')  # Plot the trajectory
                            
                else:
                    continue

                if muon.starting_frame == 0:
                        pass
                elif frame-muon.starting_frame > 10:
                    muons_c.remove(muon)
                    continue

            # Add text annotation for simulation time
            ax.annotate(f'Simulation time/s = {frame}', xy=(0.05, 0.95), xycoords='axes fraction', color='black')   

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
        # plot_detected_muons function should plot muon trajectories of tagged muons.
        # Improve RPC updating widget and RPC text files. Make this more user friendly.
        # Gaussian voltage distribution, overlap with detector strips. Threshold




