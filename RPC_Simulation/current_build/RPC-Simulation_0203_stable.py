###################################################################################################################
#The Purpose of this code is to simulate the passage and detection of Muons through a layered RPC Tracking station#
###################################################################################################################

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
import threading
from io import BytesIO
import queue

# Setting the Seaborn theme
sns.set_theme(style="darkgrid")

#Defining global variables.
global speed_of_light 
speed_of_light = 0.299792458 # m/ns

global muon_mass
muon_mass = 0.1057 #GeV/c**2

class RPC:

    def __init__(self,gas_mixture,efficiency,dimensions, strips, height,voltage, darkcount):

        #Enter gas mixture as array from GUI
        self.gas_mixture = gas_mixture
        
        #Enter empirically determined efficiency of RPC
        self.efficiency = efficiency
        
        #Dimensions of RPC in metres
        self.dimensions = dimensions
        
        #Z position of this specific RPC.
        self.height = height
        
        #Voltage applied across electrodes in the RPC, measured in kV.
        self.voltage = voltage
        
        #Configuration of strips in the RPC
        self.strips = strips
        
        #Darkcount?
        self.darkcount = darkcount
        
    def coincidence(self):
    
        pass
    
    def generate_dark_stripped(self, runtime):
        darkcountdatastripped = []

        total_dark_counts = np.random.poisson(self.darkcount * runtime * self.strips[0] * self.strips[1])
        
        for _ in range(total_dark_counts):
            x_position = np.random.uniform(0, self.dimensions[0])
            y_position = np.random.uniform(0, self.dimensions[1])
            
            x_spacing = self.dimensions[0] / (self.strips[0] - 1)
            y_spacing = self.dimensions[1] / (self.strips[1] - 1)
            x_strip = round(x_position / x_spacing) * x_spacing
            y_strip = round(y_position / y_spacing) * y_spacing
            detection_time = np.random.uniform(0, runtime)        
            
            # Append the dark count info to the darkcount list
            darkcountdatastripped.append({
                "velocity": 'Dark',
                "muon_index": 'Dark',
                "detected_x_position": x_strip,
                "detected_y_position": y_strip,
                "detected_z_position": self.height,
                "detection_time": detection_time,
                "Outcome": 'dark'
            })
        return darkcountdatastripped
    
    #Use GUI interface, generate stack of RPCs, choose gas mixtures and voltages for each. Run simulation of muon count.
        
class muon:

    def __init__(self,position,velocity,gamma,energy,theta):

        #[x,y,z] coordinates of muon
        self.position = np.array(position)

        #List of [v_x,v_y,v_z] velocity components of muon. In units of c.
        self.velocity = velocity

        #History of particle trajectories at given times.
        self.trajectory = []
        self.times = []

        #Add starting frame number for muon trajectory, useful for some plots
        self.starting_frame = 0
        
        # List containing information on coordinates and times a muon passes through an RPC plate. Y/N is whether or not the RPC registers the hit.
        # [[x,y,z,t,"Outcome"],...,[]]
        self.detected_5vector = []
        
        self.stripped_detected_5vector = []

        #This attribute is never retrieved anywhere???
        self.hits = []

        #Gamma factor of muon given it's velocity

        self.gamma = gamma
        self.energy = energy

        #Muon velocity, zenith angle.

        self.theta = theta

    def update_position(self,time_step):

        #Update the muon's current position due to its velocity.
        #Muons assumed to MIPs, such that their velocity is roughly constant over the simulation.

        #time_step is in units of nano-seconds ns.
        self.position+= np.multiply(self.velocity,speed_of_light*time_step)

    def check_hit(self,sorted_rpc_list,initial_time):

        ##Figure out dilated half-life from gamma factor.

        t_half = self.gamma*(2200) #in ns
        rate = np.log(2)/t_half

        if len(self.times)==0:
            init_time = initial_time
        else:
            init_time = self.times[0]

        times_to_rpcs = [0]
    
        for x,rpc in enumerate(sorted_rpc_list):

            success = "Y" if np.random.rand() < rpc.efficiency else "N"
            time_to_rpc = (rpc.height - max(rpc.height for rpc in sorted_rpc_list)) / (self.velocity[2]*speed_of_light) if self.velocity[2] != 0 else float('inf')
            times_to_rpcs.append(time_to_rpc)

            #ACCOUNT FOR POSSIBILITY OF MUON DECAYING
            if np.exp(-rate*(times_to_rpcs[x+1]-times_to_rpcs[x])) > np.random.rand():
                ###Muon survives until this RPC
                if 0 < self.position[0] + self.velocity[0] * time_to_rpc*speed_of_light< rpc.dimensions[0] and 0 < self.position[1] + self.velocity[1] * time_to_rpc*speed_of_light < rpc.dimensions[1]:    
                    self.detected_5vector.append([self.position[0] + self.velocity[0] * time_to_rpc*speed_of_light, self.position[1] + self.velocity[1] * time_to_rpc*speed_of_light, rpc.height, init_time + time_to_rpc, success])
                else:
                    self.detected_5vector.append([self.position[0] + self.velocity[0] * time_to_rpc*speed_of_light, self.position[1] + self.velocity[1] * time_to_rpc*speed_of_light, rpc.height, init_time + time_to_rpc, f'Missed RPC {x+1}'])
                
            else:
                ###Muon decays between previous RPC and current RPC.
                self.detected_5vector.append([float('inf'),float('inf'),float('inf'),init_time + time_to_rpc,f"Muon decayed between RPCs {x} and {x+1}"])
                #Break out of the loop, muon has decayed and can no longer be detected.
                break

    def stripped_check_hit(self, rpc_list, initial_time):
    
        if len(self.times)==0:
            init_time = initial_time
        else:
            init_time = self.times[0]

        for rpc in rpc_list:
            self.x_spacing = rpc.dimensions[0] / (rpc.strips[0] - 1)
            self.y_spacing = rpc.dimensions[1] / (rpc.strips[1] - 1)
            success = "Y" if np.random.rand() < rpc.efficiency else "N"
            time_to_rpc = (rpc.height - max(rpc.height for rpc in rpc_list)) / (self.velocity[2]*speed_of_light) if self.velocity[2] != 0 else float('inf')
            if 0 < self.position[0] + self.velocity[0] * time_to_rpc*speed_of_light < rpc.dimensions[0] and 0 < self.position[1] + self.velocity[1] * time_to_rpc*speed_of_light < rpc.dimensions[1]:
                # Calculate position at the time of potential detection
                x_pos = self.position[0] + self.velocity[0] * time_to_rpc*speed_of_light
                y_pos = self.position[1] + self.velocity[1] * time_to_rpc*speed_of_light
                # Adjust position to nearest strip point
                x_strip = round(x_pos / self.x_spacing) * self.x_spacing
                y_strip = round(y_pos / self.y_spacing) * self.y_spacing
                self.detected_5vector.append([x_strip, y_strip, rpc.height, init_time + time_to_rpc, success])
            else:
                self.detected_5vector.append([x_strip, y_strip, rpc.height, init_time + time_to_rpc, 'out'])
                    
    def simulate_path(self,rpc_list, initial_time,time_step):
        #Simulate path of muon, given time_step and initial_time in nanoseconds

        #Append initial position
        self.trajectory.append(self.position)

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
        self.style.configure('custom.TFrame', relief='solid')

        self.frame = ttk.Frame(master,style='custom.TFrame')
        self.frame.pack(fill='both', expand=True)

        # Banners and Images
        self.img = Image.open(current_directory + "\RPC_Simulation\images\Banner.png")
        self.img = self.img.resize((400,120))
        self.img_tk = ImageTk.PhotoImage(self.img)

        self.banner = ttk.Label(self.frame, image=self.img_tk)
        self.banner.grid(row=0, column=0, columnspan=2, pady=20)

        self.img2 = Image.open(current_directory + "\RPC_Simulation\images\wip.jpg")
        self.img2 = self.img2.resize((300,120))
        self.img_tk2 = ImageTk.PhotoImage(self.img2)

        self.banner2 = ttk.Label(self.frame, image=self.img_tk2)
        self.banner2.grid(row=7, column=0, columnspan=2, pady=20)

        # Buttons
        self.rpc_list = []

        self.manage_rpc_button = ttk.Button(self.frame, text="Manage RPC Plates", command=self.manage_rpc_window)
        self.manage_rpc_button.grid(row=1, column=1, pady=20)

        self.log_button = ttk.Button(self.frame, text="Save/Load RPC Log", command=self.log_rpc_window)
        self.log_button.grid(row=1, column=0, pady=20)

        self.nanoscale_sim_desc = tk.Label(self.frame, text='Simulate cosmic ray muons through setup', font=30)
        self.nanoscale_sim_desc.grid(row=2, column=0, columnspan=2, pady=10)

        self.simulation_var = tk.BooleanVar(value=True)
        self.simulation_nano_checkbox = ttk.Radiobutton(self.frame, text='Nanoseconds scale simulation', variable=self.simulation_var, value=True)
        self.simulation_nano_checkbox.grid(row=3, column=0, columnspan=2, pady=5)

        self.simulation_norm_checkbox = ttk.Radiobutton(self.frame, text='Seconds scale simulation', variable=self.simulation_var, value=False)
        self.simulation_norm_checkbox.grid(row=4, column=0, columnspan=2, pady=10)

        self.simulate_button = ttk.Button(self.frame, text="Run cosmic ray muon simulation", style='success.TButton', command=self.run_simulation)
        self.simulate_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.nanoscale_sim_desc2 = tk.Label(self.frame, text='Simulate decaying LLNP (WIP)', font=30)
        self.nanoscale_sim_desc2.grid(row=6, column=0, columnspan=2, pady=10) 
        
        #???, is this for multicore threading?
        self.queue = queue.Queue()

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
################################################################################################################### 
    def manage_rpc_window(self):
        manage_window = tk.Toplevel(self.master)
        manage_window.title("Manage RPC Plates")
        
        self.frame = ttk.Frame(manage_window)
        self.frame.pack()

        self.add_rpc_button = ttk.Button(manage_window, text="Add RPC Plate", command=self.create_rpc_window)
        self.add_rpc_button.pack(pady=5)

        self.rpc_combobox = ttk.Combobox(manage_window, state="readonly")
        self.rpc_combobox.pack(pady=20)
        self.update_rpc_combobox()

        self.edit_rpc_button = ttk.Button(manage_window,text="Edit selected RPC Plate", command = self.edit_rpc_window)
        self.edit_rpc_button.pack(pady=5)

        self.remove_rpc_button = ttk.Button(manage_window, text="Remove selected RPC Plate", command=self.remove_rpc)
        self.remove_rpc_button.pack(pady=5)
        
        self.plot_button = ttk.Button(manage_window, text="Plot the RPC Setup", command=self.plot_stations_3d)
        self.plot_button.pack(pady=5)
    
    def edit_rpc_window(self):

        edit_window = tk.Toplevel(self.master)
        selection_index = self.rpc_combobox.current()

        edit_window.title(f"Editing RPC {selection_index+1}")

        self.edit_rpc_attributes_ui(edit_window)

        save_button = ttk.Button(edit_window, text="Update RPC", command=lambda: self.update_rpc(edit_window))
        save_button.grid(row=15,column=1,pady=5)

    def create_rpc_window(self):
        rpc_window = tk.Toplevel(self.master)
        rpc_window.title("Add RPC Plate")

        # Generate UI elements for RPC attributes
        self.create_rpc_attributes_ui(rpc_window)

        save_button = ttk.Button(rpc_window, text="Save RPC", command=lambda: self.save_rpc(rpc_window))
        save_button.grid(row=15,column=1,pady=5)
    
    def remove_rpc(self):
        selection_index = self.rpc_combobox.current()
        if selection_index >= 0:  # A selection is made
            self.rpc_list.pop(selection_index)
            self.update_rpc_combobox()  # Update combobox after removal

        else:
            messagebox.showwarning(title="Warning",message="No RPC currently selected")
            return

    def show_entry(self, var, widget):
        #Decide whether or not to enable a widget passed on a user checkbox
        if var.get():
            widget.configure(state='normal')
        else:
            widget.configure(state='disabled')

    def update_rpc_combobox(self):

        rpc_descriptions = [f"RPC {idx+1}" for idx,rpc in enumerate(self.rpc_list)]
        self.rpc_combobox['values'] = rpc_descriptions

        if rpc_descriptions:
            self.rpc_combobox.current(0)
        else:
            self.rpc_combobox.set('')
    
    def create_rpc_attributes_ui(self, rpc_window):

        #UI Elements for entering the attributes of the RPC being added.

        # Height of RPC
        self.height_var_label = ttk.Label(rpc_window, text="Z axis (in metres) of the RPC plate: ")
        self.height_var_label.grid(row=0, column=0, pady=5)
        self.height_var = tk.DoubleVar()
        self.height_var_entry = ttk.Entry(rpc_window, textvariable=self.height_var)
        self.height_var_entry.grid(row=0, column=1, pady=5)

        # Voltage across the RPC plate in kV
        self.voltage_var_label = ttk.Label(rpc_window, text="Voltage applied across the RPC electrode (kV): ")
        self.voltage_var_label.grid(row=1, column=0, pady=5)
        self.voltage_var = tk.DoubleVar()
        self.voltage_var_entry = ttk.Entry(rpc_window, textvariable=self.voltage_var)
        self.voltage_var_entry.grid(row=1, column=1, pady=5)

        # Dimensions of RPC (assumed rectangular)
        self.x_var_label = ttk.Label(rpc_window, text="Width of RPC (m): ")
        self.x_var_label.grid(row=2, column=0, pady=5)
        self.x_var = tk.DoubleVar()
        self.x_var_entry = ttk.Entry(rpc_window, textvariable=self.x_var)
        self.x_var_entry.grid(row=2, column=1, pady=5)

        self.y_var_label = ttk.Label(rpc_window, text="Length of RPC (m): ")
        self.y_var_label.grid(row=3, column=0, pady=5)
        self.y_var = tk.DoubleVar()
        self.y_var_entry = ttk.Entry(rpc_window, textvariable=self.y_var)
        self.y_var_entry.grid(row=3, column=1, pady=5)

        self.t_var_label = ttk.Label(rpc_window, text="Thickness of RPC (mm): ")
        self.t_var_label.grid(row=4, column=0, pady=5)
        self.t_var = tk.DoubleVar()
        self.t_var_entry = ttk.Entry(rpc_window, textvariable=self.t_var)
        self.t_var_entry.grid(row=4, column=1, pady=5)
            
        self.xs_var_label = ttk.Label(rpc_window, text="Number of strips in x direction: ")
        self.xs_var_label.grid(row=5, column= 0, pady = 5)
        self.xs_var = tk.DoubleVar()
        self.xs_var_entry = ttk.Entry(rpc_window, textvariable=self.xs_var)
        self.xs_var_entry.grid(row=5, column= 1, pady=5 )
        
        self.ys_var_label = ttk.Label(rpc_window, text="Number of strips in y direction: ")
        self.ys_var_label.grid(row=6, column= 0, pady = 5)
        self.ys_var = tk.DoubleVar()
        self.ys_var_entry = ttk.Entry(rpc_window, textvariable=self.ys_var)
        self.ys_var_entry.grid(row=6, column= 1, pady = 5)
        
        self.darkcount_label = ttk.Label(rpc_window, text="Darkcount rate /ns: ")
        self.darkcount_label.grid(row=7, column= 0, pady = 5)
        self.darkcount = tk.DoubleVar()
        self.darkcount_entry = ttk.Entry(rpc_window, textvariable=self.darkcount)
        self.darkcount_entry.grid(row=7, column= 1, pady = 5)
        
        self.gas_desc = tk.Label(rpc_window, text='% Of Gas mixture by volume: ',font = 5)
        self.gas_desc.grid(row=8,column=1, pady=2)

        # Gas mixture of RPC
        self.gases = ["Isobutane", "Argon", "CO2", "N2"]
        self.selected_gases = {gas: [tk.BooleanVar(), tk.DoubleVar()] for gas in self.gases}

        i = 9
    
        for gas in self.gases:

            gas_percentage_var = tk.DoubleVar()  # Use local variable instead of self.gas_percentage_var
            gas_percentage_entry = ttk.Entry(rpc_window, textvariable=gas_percentage_var, state="disabled")
            gas_percentage_entry.grid(row=i, column= 1, pady = 5)

            # Checkbox
            select_gas = tk.BooleanVar()  # Use local variable instead of self.select_gas
            chk = ttk.Checkbutton(rpc_window, text=gas, variable=select_gas, command=lambda v=select_gas, e=gas_percentage_entry: self.show_entry(v, e))
            chk.grid(row=i, column= 0, pady = 5)

            # Add gas selection to dictionary when checkbox is clicked
            self.selected_gases[gas] = (select_gas, gas_percentage_var)  # Store variables, not their current values

            i+=1
                    
        # Efficiency of RPC
        self.efficiency_var_label = ttk.Label(rpc_window, text="Hit efficiency of the RPC: ")
        self.efficiency_var_label.grid(row=14, column= 0, pady = 5)
        self.efficiency_var = tk.DoubleVar()
        self.efficiency_var_entry = ttk.Entry(rpc_window, textvariable=self.efficiency_var)
        self.efficiency_var_entry.grid(row=14, column= 1, pady = 5)
    
    def edit_rpc_attributes_ui(self, edit_window):

        selection_index = self.rpc_combobox.current()
        current_rpc = self.rpc_list[selection_index]

        current_voltage = float(current_rpc.voltage) if current_rpc.voltage else 0.0
        current_efficiency = float(current_rpc.efficiency) if current_rpc.efficiency else 0.0
        current_height = float(current_rpc.height) if current_rpc.height else 0.0
        current_x = float(current_rpc.dimensions[0]) if current_rpc.dimensions[0] else 0.0
        current_y = float(current_rpc.dimensions[1]) if current_rpc.dimensions[1] else 0.0
        current_t = float(current_rpc.dimensions[2]*1000) if current_rpc.dimensions[2] else 0.0 
        current_darkcount = float(current_rpc.darkcount) if current_rpc.darkcount else 0.0
        current_x_strips = int(current_rpc.strips[1]) if current_rpc.strips[0] else 0
        current_y_strips = int(current_rpc.strips[0]) if current_rpc.strips[1] else 0

        #UI Elements for entering the attributes of the RPC being added.

        # Height of RPC
        self.height_var_label = ttk.Label(edit_window, text="Z axis (in metres) of the RPC plate: ")
        self.height_var_label.grid(row=0, column= 0, pady = 5)
        self.height_var = tk.DoubleVar()
        self.height_var.set(current_height)
        self.height_var_entry = ttk.Entry(edit_window, textvariable=self.height_var)
        self.height_var_entry.grid(row=0, column= 1, pady = 5)
        
        # Voltage across the RPC plate in kV
        self.voltage_var_label = ttk.Label(edit_window, text="Voltage applied across the RPC electrode (kV): ")
        self.voltage_var_label.grid(row=1, column= 0, pady = 5)
        self.voltage_var = tk.DoubleVar()
        self.voltage_var.set(current_voltage)
        self.voltage_var_entry = ttk.Entry(edit_window, textvariable=self.voltage_var)
        self.voltage_var_entry.grid(row=1, column= 1, pady = 5)
        
        # Dimensions of RPC (assumed rectangular)
        self.x_var_label = ttk.Label(edit_window, text="Width of RPC (m): ")
        self.x_var_label.grid(row=2, column= 0, pady = 5)
        self.x_var = tk.DoubleVar()
        self.x_var.set(current_x)
        self.x_var_entry = ttk.Entry(edit_window, textvariable=self.x_var)
        self.x_var_entry.grid(row=2, column= 1, pady = 5)
        
        self.y_var_label = ttk.Label(edit_window, text="Length of RPC (m): ")
        self.y_var_label.grid(row=3, column= 0, pady = 5)
        self.y_var = tk.DoubleVar()
        self.y_var.set(current_y)
        self.y_var_entry = ttk.Entry(edit_window, textvariable=self.y_var)
        self.y_var_entry.grid(row=3, column= 1, pady = 5)

        self.t_var_label = ttk.Label(edit_window, text="Thickness of RPC (mm): ")
        self.t_var_label.grid(row=4, column= 0, pady = 5)
        self.t_var = tk.DoubleVar()
        self.t_var.set(current_t)
        self.t_var_entry = ttk.Entry(edit_window, textvariable=self.t_var)
        self.t_var_entry.grid(row=4, column= 1, pady = 5)
        
        self.xs_var_label = ttk.Label(edit_window, text="Number of strips in x direction: ")
        self.xs_var_label.grid(row=5, column= 0, pady = 5)
        self.xs_var = tk.DoubleVar()
        self.xs_var.set(current_x_strips)
        self.xs_var_entry = ttk.Entry(edit_window, textvariable=self.xs_var)
        self.xs_var_entry.grid(row=5, column= 1, pady = 5)
        
        self.ys_var_label = ttk.Label(edit_window, text="Number of strips in y direction: ")
        self.ys_var_label.grid(row=6, column= 0, pady = 5)
        self.ys_var = tk.DoubleVar()
        self.ys_var.set(current_y_strips)
        self.ys_var_entry = ttk.Entry(edit_window, textvariable=self.ys_var)
        self.ys_var_entry.grid(row=6, column= 1, pady = 5)
        
        self.darkcount_label = ttk.Label(edit_window, text="Darkcount rate /ns: ")
        self.darkcount_label.grid(row=7, column= 0, pady = 5)
        self.darkcount = tk.DoubleVar()
        self.darkcount.set(current_darkcount)
        self.darkcount_entry = ttk.Entry(edit_window, textvariable=self.darkcount)
        self.darkcount_entry.grid(row=7, column= 1, pady = 5)

        self.gas_desc = tk.Label(edit_window, text='% Of Gas mixture by volume: ',font = 5)
        self.gas_desc.grid(row=8,column=1, pady=2)
        
        #current_gas_mixture = float(current_rpc.gas_mixture) if current_rpc.gas_mixture else {}

        # Gas mixture of RPC
        self.gases = ["Isobutane", "Argon", "CO2", "N2"]

        if current_rpc.gas_mixture:
            self.selected_gases = current_rpc.gas_mixture
        else:
            self.selected_gases = {gas: [tk.BooleanVar(), tk.DoubleVar()] for gas in self.gases}

        i = 9
    
        for gas in self.gases:

            if gas in self.selected_gases:
                gas_percentage_var = tk.DoubleVar()  # Use local variable instead of self.gas_percentage_var
                gas_percentage_var.set(self.selected_gases[gas])
                gas_percentage_entry = ttk.Entry(edit_window, textvariable=gas_percentage_var, state="normal")
                gas_percentage_entry.grid(row=i, column= 1, pady = 5)

                # Checkbox
                select_gas = tk.BooleanVar()  # Use local variable instead of self.select_gas
                select_gas.set(True)
                chk = ttk.Checkbutton(edit_window, text=gas, variable=select_gas, command=lambda v=select_gas, e=gas_percentage_entry: self.show_entry(v, e))
                chk.grid(row=i, column= 0, pady = 5)

                # Add gas selection to dictionary when checkbox is clicked
                self.selected_gases[gas] = (select_gas, gas_percentage_var)  # Store variables, not their current values

            else:
            
                gas_percentage_var = tk.DoubleVar()  # Use local variable instead of self.gas_percentage_var
                gas_percentage_entry = ttk.Entry(edit_window, textvariable=gas_percentage_var, state="disabled")
                gas_percentage_entry.grid(row=i, column= 1, pady = 5)

                # Checkbox
                select_gas = tk.BooleanVar()  # Use local variable instead of self.select_gas
                chk = ttk.Checkbutton(edit_window, text=gas, variable=select_gas, command=lambda v=select_gas, e=gas_percentage_entry: self.show_entry(v, e))
                chk.grid(row=i, column= 0, pady = 5)

                # Add gas selection to dictionary when checkbox is clicked
                self.selected_gases[gas] = (select_gas, gas_percentage_var)  # Store variables, not their current values

            i+=1
                    
        # Efficiency of RPC
        self.efficiency_var_label = ttk.Label(edit_window, text="Hit efficiency of the RPC: ")
        self.efficiency_var_label.grid(row=14, column= 0, pady = 5)
        self.efficiency_var = tk.DoubleVar()
        self.efficiency_var.set(current_efficiency)
        self.efficiency_var_entry = ttk.Entry(edit_window, textvariable=self.efficiency_var)
        self.efficiency_var_entry.grid(row=14, column= 1, pady = 5)

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
        darkcount = float(self.darkcount.get())
        new_rpc = RPC(height=height, efficiency=efficiency,
                        dimensions=dimensions,strips = strips, voltage=voltage, gas_mixture=gas_mixture, darkcount=darkcount)
        

        # Add RPC object to the array
        self.rpc_list.append(new_rpc)

        self.update_rpc_combobox()

        # Close the RPC window
        rpc_window.destroy()

    def update_rpc(self, edit_window):

        selection_index = self.rpc_combobox.current()

        # Get user inputs and create RPC object

        height = float(self.height_var.get())
        voltage = float(self.voltage_var.get())
        #dimensions in metres.
        dimensions = [float(self.x_var.get()), float(self.y_var.get()),float(self.t_var.get())/1000]
        strips = [int(self.xs_var.get()), int(self.ys_var.get())]
        efficiency = float(self.efficiency_var.get())
        gas_mixture = {gas: percentage.get() for gas, (selected, percentage) in self.selected_gases.items() if selected.get()}
        darkcount = float(self.darkcount.get())
        new_rpc = RPC(height=height, efficiency=efficiency,
                        dimensions=dimensions,strips = strips, voltage=voltage, gas_mixture=gas_mixture, darkcount=darkcount)
        
        # Update RPC object in the array
        self.rpc_list[selection_index] = new_rpc
        
        self.update_rpc_combobox()

        # Close the RPC window
        edit_window.destroy()

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
                            f"Darkcount Rate = {rpc.darkcount},\n" \
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
                                efficiency=float(filt_item[7]), gas_mixture=eval(filt_item[8]), darkcount=float(filt_item[9]))
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
                                efficiency=float(filt_item[7]), gas_mixture=eval(filt_item[8]), darkcount=float(filt_item[9]))
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
        self.muon_flux_var.set(0.01)
        self.muon_flux_entry = ttk.Entry(simulation_window, textvariable=self.muon_flux_var)
        self.muon_flux_entry.pack(pady=5)

        # Simulation time in ns
        self.sim_time_label = ttk.Label(simulation_window, text="Simulation time (ns):")
        self.sim_time_label.pack(pady=5)
        self.sim_time_var = tk.DoubleVar()
        self.sim_time_var.set(100)
        self.sim_time_entry = ttk.Entry(simulation_window, textvariable=self.sim_time_var)
        self.sim_time_entry.pack(pady=5)

        #check if you need to simulate Path
        self.use_paths_var = tk.BooleanVar()
        self.use_paths_check = ttk.Checkbutton(simulation_window, text="Use paths", variable=self.use_paths_var)
        self.use_paths_check.pack(pady=5)

        # Checkbox for using strips
        self.use_strips_var = tk.BooleanVar()
        self.use_strips_check = ttk.Checkbutton(simulation_window, text="Use strips", variable=self.use_strips_var)
        self.use_strips_check.pack(pady=5)

        #Add checkbox for enabling dark counts
        self.use_darkcount_var = tk.BooleanVar()
        self.use_darkcount_check = ttk.Checkbutton(simulation_window, text="Use darkcount", variable=self.use_darkcount_var)
        self.use_darkcount_check.pack(pady=5)
        
        # Start simulation button combined
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation_combinednano)
        self.start_sim_button.pack(pady=5)
    
    def generate_muon_at_time(self,theta,h,energy):

        E = energy #GeV
        gamma = E / muon_mass
        beta = np.sqrt(1-1/(gamma**2))
        # beta = 0.98
        # gamma = np.sqrt(1/(1-beta**2))

        #Could alternate phi such that it always points towards atleast one RPC.
        phi = np.random.uniform(0, 2 * np.pi)
        
        velocity = np.multiply(beta,[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)])
        time_of_travel = np.abs(h / velocity[2])

        #This feels wrong, will change soon....
        
        extension = np.multiply(velocity, time_of_travel)
        position = [np.random.uniform(-extension[0],max(rpc.dimensions[0] for rpc in self.rpc_list)-extension[0]),np.random.uniform(-extension[1],max(rpc.dimensions[1] for rpc in self.rpc_list)-extension[1]) , max(rpc.height for rpc in self.rpc_list)]
        
        return muon(position= position, velocity= velocity, gamma = gamma, energy=E, theta= theta)

    def start_simulation_combinednano(self):

        """Function to start the simulation using parameters from the GUI."""

        def energy_dist(E):
            #E In units of GeV
            #Parameterise the distribution.

            E_0 = 4.29
            eps = 854
            n = 3.01

            #Energy dist from paper.
            p = ((E_0+E)**(-n))* ((1+ E / eps)**(-1))
        
            return p

        total_sim_time = self.sim_time_var.get()
        detected_muons = []
        detected_dark_muons = pd.DataFrame({            
            "velocity": [np.nan],
            "muon_index": [np.nan],
            "detected_x_position": [np.nan],
            "detected_y_position": [np.nan],
            "detected_z_position": [np.nan],
            "detection_time": [np.nan],
            "Outcome": [np.nan]
            })
        muons = []
        sim_time = 0
        muon_index = 0

        #Muon flux, muon_flux_var is measured in /cm^2/ns
        muons_flux = self.muon_flux_var.get()
        #Now calculate the expected muon rate given the dimensions of the problem.
        area_m2 = max(rpc.dimensions[0] for rpc in self.rpc_list)*max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1025
        #Now calculate the average rate of muon arrival given the problem specifics.    
        rate = muons_flux*area_m2*(1e4)
        
        traj_time_step = min(rpc.dimensions[2] for rpc in self.rpc_list) / (0.299792458)

        def generate_theta():
            #GENERATE MUON ZENITH ANGLE FROM MC Accept/Reject algorithm.
            def pdf(x):
                return 4/np.pi * np.cos(x)**2
            
            while True:
                theta = np.random.uniform(0,np.pi/2)
                p = np.random.uniform(0,pdf(0))
                
                if p < pdf(theta):
                    theta_generated = theta
                    break
                    
            return theta_generated

        #Sample energy from distribution.

        #sampling 0.01 above muon_mass to avoid infinite time between rpcs.
        energy_vals = np.linspace(muon_mass+0.01,300,10000)
        energy_probs = [energy_dist(x) for x in energy_vals]
        norm_energy_probs = np.multiply(1/(np.sum(energy_probs)),energy_probs)

        #Calculate necessary parameters outside of while loop.
        max_z = max(rpc.height for rpc in self.rpc_list)
        min_z = min(rpc.height for rpc in self.rpc_list)
        h = max_z - min_z

        def sort_func(x):
            return x.height
        
        self.rpc_list.sort(key=sort_func,reverse=True)
        sorted_rpc_list = self.rpc_list

        while sim_time < total_sim_time:

            time_to_next_muon = -np.log(1-np.random.uniform()) / rate
            sim_time += time_to_next_muon
            if sim_time > total_sim_time:
                break
                
            theta = generate_theta()
            energy = np.random.choice(energy_vals, p=norm_energy_probs)
            muon_instance = self.generate_muon_at_time(theta=theta,h=h,energy = energy)
            
            if self.use_paths_var.get() == True:        
                muon_instance.simulate_path(self.rpc_list, sim_time, traj_time_step)
            
            if self.use_strips_var.get() == True:
                muon_instance.stripped_check_hit(self.rpc_list, initial_time = sim_time)
            else:
                muon_instance.check_hit(sorted_rpc_list,initial_time = sim_time)
                 
            for x in muon_instance.detected_5vector:
                detected_muons.append({
                    "velocity": muon_instance.velocity,
                    "muon_index": muon_index,
                    "detected_x_position":x[0],
                    "detected_y_position":x[1],
                    "detected_z_position": x[2],
                    "detection_time": x[3],
                    "Outcome":x[4],
                    "Energy/GeV":muon_instance.energy,

                        })
                
            muon_index += 1
            muons.append(muon_instance)

                
        if self.use_darkcount_var.get() == True:
            for rpc in self.rpc_list:  
                dark = pd.DataFrame(RPC.generate_dark_stripped(rpc, total_sim_time * 1e-9))
                detected_dark_muons = pd.concat([dark, detected_dark_muons], ignore_index=True)            
                
        df_detected_muons = pd.DataFrame(detected_muons)
        if self.use_darkcount_var.get() == True:
            df_detected_muons = pd.concat([detected_dark_muons, df_detected_muons], ignore_index=True)
            
        self.simulation_finished_dialog(df_detected_muons,muons)

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
        self.muon_flux_var.set(0.017) #Ground level muon flux cm^2/s
        self.muon_flux_entry = ttk.Entry(simulation_window, textvariable=self.muon_flux_var)
        self.muon_flux_entry.pack(pady=5)

        # Simulation time in s
        self.sim_time_label = ttk.Label(simulation_window, text="Simulation time (s):")
        self.sim_time_label.pack(pady=5)
        self.sim_time_var = tk.DoubleVar()
        self.sim_time_var.set(100)
        self.sim_time_entry = ttk.Entry(simulation_window, textvariable=self.sim_time_var)
        self.sim_time_entry.pack(pady=5)

        #check if you need to simulate Path
        self.use_paths_var = tk.BooleanVar(value=False)
        self.use_paths_check = ttk.Checkbutton(simulation_window, text="Use paths", variable=self.use_paths_var)
        self.use_paths_check.pack(pady=5)

        # Checkbox for using strips
        self.use_strips_var = tk.BooleanVar()
        self.use_strips_check = ttk.Checkbutton(simulation_window, text="Use strips", variable=self.use_strips_var)
        self.use_strips_check.pack(pady=5)

        #Add checkbox for enabling dark counts
        self.use_darkcount_var = tk.BooleanVar()
        self.use_darkcount_check = ttk.Checkbutton(simulation_window, text="Use darkcount", variable=self.use_darkcount_var)
        self.use_darkcount_check.pack(pady=5)

        # Start simulation button
        self.start_sim_button = ttk.Button(simulation_window, text="Start Simulation", command=self.start_simulation_normscale)
        self.start_sim_button.pack(pady=5)
               
    def start_simulation_normscale(self):

        sim_time = self.sim_time_var.get() #Simulation time in seconds
        detected_muons = []
        muons = []
        muon_index = 0
        running_time = 0 #Start counting time

        def energy_dist(E):
            #E In units of GeV
            #Parameterise the distribution.

            E_0 = 4.29
            eps = 854
            n = 3.01

            #Energy dist from paper.
            p = ((E_0+E)**(-n))* ((1+ E / eps)**(-1))
        
            return p

        energy_vals = np.linspace(muon_mass+0.01,300,10000) #sampling 0.01 above muon_mass to avoid infinite time between rpcs.
        energy_probs = [energy_dist(x) for x in energy_vals]
        norm_energy_probs = np.multiply(1/(np.sum(energy_probs)),energy_probs)

        muons_flux = self.muon_flux_var.get() #Muon flux, muon_flux_var is measured in /cm^2/s
        area_m2 = max(rpc.dimensions[0] for rpc in self.rpc_list)*max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1025  
        rate = muons_flux*area_m2*(1e4) #Rate /s

        def generate_theta():
            #GENERATE MUON ZENITH ANGLE FROM MC Accept/Reject algorithm.
            def pdf(x):
                return 4/np.pi * np.cos(x)**2
            
            while True:
                theta = np.random.uniform(0,np.pi/2)
                p = np.random.uniform(0,pdf(0))
                
                if p < pdf(theta):
                    theta_generated = theta
                    break
                    
            return theta_generated

        traj_time_step = min(rpc.dimensions[2] for rpc in self.rpc_list) / (0.299792458)

        #Calculate necessary parameters outside of while loop.
        max_z = max(rpc.height for rpc in self.rpc_list)
        min_z = min(rpc.height for rpc in self.rpc_list)
        h = max_z - min_z

        def sort_func(x):
            return x.height
        
        self.rpc_list.sort(key=sort_func,reverse=True)
        sorted_rpc_list = self.rpc_list

        while running_time < sim_time:

            #Instead of simulating each time step, I am now drawing the time t of the next muon hit from a probability distribution.
            #The distribution used here is a first passage time distribution. For a poisson process this leads to an exponential distribution.
            #Due to the exponential nature of this first passage time distribution, it makes it difficult to sample discretely from it.
            #Have to use inverse transform sampling from the first passage time distribution instead.

            u = np.random.uniform() #Luckily np.random.uniform() excludes 1, as this produces an infinite result from inverse transform sampling...
            muon_time = -1 / rate *np.log(1-u)  #muon_time here is the time between when the last muon was produced and the next muon is generated.
            running_time+= muon_time #running_time (in seconds) now updated to time when new muon is generated

            if running_time >= sim_time: #it is possible for running_time to exceed sim_time in the while loop now, just including this break to avoid this.
                break
            
            theta = generate_theta()
            energy = np.random.choice(energy_vals, p=norm_energy_probs)
            muon_instance = self.generate_muon_at_time(theta=theta,h=h,energy = energy) 

            if self.use_paths_var.get() == True:   
                muon_instance.simulate_path(self.rpc_list, initial_time=(running_time*1e9), time_step=traj_time_step) 
            if self.use_strips_var.get() == True:
                muon_instance.stripped_check_hit(self.rpc_list, initial_time=sim_time)
            else:
                muon_instance.check_hit(sorted_rpc_list, initial_time=(running_time*1e9))

            for x in muon_instance.detected_5vector:
                detected_muons.append({
                "velocity": muon_instance.velocity,
                "muon_index": muon_index,
                "detected_x_position":x[0],
                "detected_y_position":x[1],
                "detected_z_position": x[2],
                "detection_time": x[3] * 10 ** -9,
                "Outcome":x[4],
                "Energy/GeV": muon_instance.energy,
                    })
            muon_index += 1
            muons.append(muon_instance)
    
        df_detected_muons = pd.DataFrame(detected_muons)
        
        detected_muons = []
        detected_dark_muons = pd.DataFrame({            
            "velocity": [np.nan],
            "muon_index": [np.nan],
            "detected_x_position": [np.nan],
            "detected_y_position": [np.nan],
            "detected_z_position": [np.nan],
            "detection_time": [np.nan],
            "Outcome": [np.nan]
            })
    
        if self.use_darkcount_var.get() == True:
            for rpc in self.rpc_list:  
                dark = pd.DataFrame(RPC.generate_dark_stripped(rpc, sim_time))
                detected_dark_muons = pd.concat([dark, detected_dark_muons], ignore_index=True)            
            df_detected_muons = pd.concat([detected_dark_muons, df_detected_muons], ignore_index=True)
            
        self.simulation_finished_dialog(df_detected_muons,muons)

###################################################################################################################
###################################################################################################################
#Simulation result section
###################################################################################################################
    def simulation_finished_dialog(self, df_selected_muons,muons):

        dialog_window = tk.Toplevel(self.master)
        dialog_window.title(f"Simulation Finished")

        dialog_window_desc = tk.Label(dialog_window, text=f'{len(muons)} muons generated')
        dialog_window_desc.pack(padx=30, pady=30)

        view_data_button = ttk.Button(dialog_window, text="View Data", command=lambda: self.view_data(df_selected_muons))
        view_data_button.pack(pady=5)

        # Button to plot data on 3D plot
        plot_data_button = ttk.Button(dialog_window, text="Plot Muon Distributions", command=lambda: self.plot_muon_distributions(df_selected_muons,muons))
        plot_data_button.pack(pady=5)

        # Button to save data into a CSV
        save_data_button = ttk.Button(dialog_window, text="Save Data Again", command=lambda: self.save_data_again(df_selected_muons))
        save_data_button.pack(pady=5)
        
        play_video_button = ttk.Button(dialog_window, text="Play Video",command=lambda: self.play_video(muons,df_selected_muons))
        play_video_button.pack(pady=5)

        if self.use_paths_var.get():
            s='normal'
        else:
            s='disabled'

        self.video_plot_paths_var = tk.BooleanVar(value=False)
        self.video_plot_paths_var_checkbox = ttk.Checkbutton(dialog_window, text='Plot the trajectories in the plot?', state=s,variable = self.video_plot_paths_var)
        self.video_plot_paths_var_checkbox.pack(anchor='center')
        
    def play_video(self,muons,df_selected_muons):

        #Select video player format if nanoscale or norm scale simulation.
        if self.simulation_var.get():
            self.play_video_nano(muons,df_selected_muons)
        else:
            self.play_video_norm(muons,df_selected_muons)

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

    def plot_muon_distributions(self, df_selected_muons,muons):

        muon_distributions_window = tk.Toplevel(self.master)
        muon_distributions_window.title("Plot muon distributions")

        self.plot_energy_distribution_button = ttk.Button(muon_distributions_window,text="Plot muon energy distribution", command=lambda: self.plot_muon_energy_distribution(df_selected_muons,muons))
        self.plot_energy_distribution_button.grid(row=0,column=0,pady=5,padx=20)

        self.plot_zenith_distribution_button = ttk.Button(muon_distributions_window,text="Plot muon zenith angle distribution", command=lambda: self.plot_zenith_angle_distribution(df_selected_muons,muons))
        self.plot_zenith_distribution_button.grid(row=1,column=0,pady=5,padx=20)

    def plot_muon_energy_distribution(self,df_selected_muons,muons):

        energies = [x.energy for x in muons if x.energy<50] #energies in GeV
        num_points = len(energies)

        def energy_dist(E):
            #E In units of GeV
            #Parameterise the distribution.
            E_0 = 4.29
            eps = 854
            n = 3.01
            #Energy dist from literature.
            p = ((E_0+E)**(-n))* ((1+ E / eps)**(-1))
            return p
        
        energy_vals = np.linspace(0,300,100000)
        energy_probs = np.array([energy_dist(x) for x in energy_vals])
        norm_energy_probs = energy_probs / np.sum(energy_probs)

        cdf = np.cumsum(norm_energy_probs)
        cdf_spacing = 300 / 1e5

        plt.figure()

        #Plot histogram of data.
        num_bins = 100  # Adjust the number of bins as needed
        counts, bin_edges, _= plt.hist(energies, bins=num_bins, alpha=0.7, label='Simulated muon distribution')
        
        bin_midpoints = []
        midpoint_freq = []

        bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

        for j in range(len(bin_edges)-1):
            start_bin_index = int(np.floor(bin_edges[j]/ cdf_spacing))
            end_bin_index = int(np.floor(bin_edges[j+1]/ cdf_spacing))
            cum_prob = cdf[end_bin_index]-cdf[start_bin_index]
            freq = cum_prob * num_points
            midpoint_freq.append(freq)

        #Plot distribution from literature.
        plt.plot(bin_midpoints,midpoint_freq,c='red',label='Generating distribution')

        plt.xlabel("Energy/GeV")
        plt.ylabel("Frequency")
        plt.legend()
        plt.xlim(left=0)
        plt.title("Muon energy distribution")
        plt.show()

    def plot_zenith_angle_distribution(self,df_selected_muons,muons):
        
        generated_theta_vals = [x.theta for x in muons]
        num_points = len(generated_theta_vals)

        theta_vals = np.linspace(0,np.pi/2,100000,endpoint=False)
        probs = [4/(np.pi) * (np.cos(x))**2 for x in theta_vals]
        norm_probs = np.multiply(1/(np.sum(probs)),probs)

        cdf = np.cumsum(norm_probs)
        cdf_spacing = np.pi/2 / 1e5

        plt.figure()

        #Plot histogram of data.
        num_bins = 100  # Adjust the number of bins as needed
        counts, bin_edges, _= plt.hist(generated_theta_vals, bins=num_bins, alpha=0.7, label='Simulated muon distribution')
        
        bin_midpoints = []
        midpoint_freq = []

        bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

        for j in range(len(bin_edges)-1):
            start_bin_index = int(np.floor(bin_edges[j]/ cdf_spacing))
            end_bin_index = int(np.floor(bin_edges[j+1]/ cdf_spacing))
            cum_prob = cdf[end_bin_index]-cdf[start_bin_index]
            freq = cum_prob * num_points
            midpoint_freq.append(freq)

        #Plot distribution from literature.
        plt.plot(bin_midpoints,midpoint_freq,c='red',label='Generating distribution')

        plt.xlabel("Zenith Angle/ radians")
        plt.ylabel("Frequency")

        plt.legend()
        plt.xlim(left=0)
        plt.title("Muon zenith angle distribution")
        plt.show()

    def save_data_again(self, muons):
        filepath = filedialog.asksaveasfilename(defaultextension="csv", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            muons.to_csv(filepath, index=False)
            messagebox.showinfo("Data Saved", "The muons data has been saved to " + filepath)
            
    def play_video_nano(self,muons,df):

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize empty arrays to store accumulated positions
        x_accumulated, y_accumulated, z_accumulated = [], [], []
        
        for rpc in self.rpc_list:
            z = rpc.height
            width, length, _ = rpc.dimensions

            vertices = np.array([[0, 0, z],
                                [width, 0, z],
                                [width, length, z],
                                [0, length, z]])
            
            # Define the vertices of the rectangle
            faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
            poly3d = Poly3DCollection(faces, alpha=0.01, edgecolors='r', linewidths=1)
            ax.add_collection3d(poly3d)

        #Simulation time in nanoseconds
        sim_time = self.sim_time_var.get()

        number_of_frames = int(sim_time)+1

        def update(frame, x_accumulated = x_accumulated, y_accumulated = y_accumulated,  z_accumulated = z_accumulated ):
                
            #If you would like to keep the trajectory plotted, then just remove ax.cla().
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
                poly3d = Poly3DCollection(faces, alpha=0.01, edgecolors='r', linewidths=1, facecolors='cyan')
                ax.add_collection3d(poly3d)

            if self.video_plot_paths_var.get():

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
                    else:
                        continue

                scat = ax.scatter([],[],[],alpha=1,c="red")

                for rpc in self.rpc_list:
                    z = rpc.height
                    width, length, _ = rpc.dimensions

                    vertices = np.array([[0, 0, z],
                                        [width, 0, z],
                                        [width, length, z],
                                        [0, length, z]])
                    
                    # Define the vertices of the rectangle
                    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
                    poly3d = Poly3DCollection(faces, alpha=0.2, edgecolors='r', linewidths=1)
                    ax.add_collection3d(poly3d)

                # Filter data for the cumulative frame.
                # 1 frame = 1 ns

                current_data = df[(frame-1 <= df['detection_time']) & (df['detection_time'] <= frame)]
                x_current = current_data['detected_x_position'].values
                y_current = current_data['detected_y_position'].values
                z_current = current_data['detected_z_position'].values

                #SOMETHING IS WRONG WITH z_current???? eg gives wrong value, plot it and see

                # Accumulate the positions
                x_accumulated.extend(x_current)
                y_accumulated.extend(y_current)
                z_accumulated.extend(z_current)
                if len(x_accumulated) > 40:
                    x_accumulated = x_accumulated[-40:]
                if len(y_accumulated) > 40:
                    y_accumulated = y_accumulated[-40:]
                if len(z_accumulated) > 40:
                    z_accumulated = z_accumulated[-40:]
                
                # Update scatter plot data
                scat._offsets3d = (x_accumulated, y_accumulated, z_accumulated)

                # Add text annotation for simulation time
                ax.annotate(f'Simulation time/ns = {frame}', xy=(0.05, 0.95), xycoords='axes fraction', color='black')

                if  frame == number_of_frames-1:
                    x_accumulated.clear()
                    y_accumulated.clear()
                    z_accumulated.clear()
                        # Also clear the scatter plot
                    scat.remove()
                    # Recreate the scatter plot
                    scat = ax.scatter([], [], [])
                    return scat, ax
                else:
                    return scat, ax
            
            else:

                scat = ax.scatter([],[],[])
                ax.annotate(f'Simulation time/ns = {frame}', xy=(0.05, 0.95), xycoords='axes fraction', color='black')

                # Filter data for the cumulative frame.
                # 1 frame = 1 ns
                        # Do we really want equality for both conditions here?
                current_data = df[(frame-1 <= df['detection_time']) & (df['detection_time'] <= frame)]
                x_current = current_data['detected_x_position'].values
                y_current = current_data['detected_y_position'].values
                z_current = current_data['detected_z_position'].values
                # Assuming 'width' and 'length' are defined elsewhere in your code
                conditionx = x_current <= width
                conditiony = y_current <= length
                conditionx1 = x_current >= 0
                conditionx2 = y_current >= 0

                # Combine the conditions using logical AND to ensure we only keep entries that satisfy both
                combined_condition = conditionx & conditiony & conditionx1 & conditionx2

                # Apply the combined condition to all arrays
                x_filtered = x_current[combined_condition]
                y_filtered = y_current[combined_condition]
                z_filtered = z_current[combined_condition] # Now applying filtering to z_current as well

                # Now, all filtered arrays will have the same dimension
                x_accumulated.extend(x_filtered)
                y_accumulated.extend(y_filtered)
                z_accumulated.extend(z_filtered)
                scat._offsets3d = (x_accumulated, y_accumulated, z_accumulated)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(-max(rpc.dimensions[0] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.1)
                ax.set_ylim(-max(rpc.dimensions[1] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1)
                ax.set_zlim(0, max(rpc.height for rpc in self.rpc_list) + 2)
                
                
                muon_detection_times = df.groupby('muon_index')['detection_time'].agg(['min', 'max']).to_dict('index')
                #Again do we really want equality for both conditions here?
                relevant_muons = [index for index, times in muon_detection_times.items() if times['min'] <= frame <= times['max']]
                if relevant_muons:
                    # Filter dataframe for relevant muons.
                    current_data_line = df[df['muon_index'].isin(relevant_muons) & ((df['Outcome'] == 'Y') | (df['Outcome'] == 'N'))]
                    current_data_line = current_data_line.dropna(subset=['detected_x_position', 'detected_y_position', 'detected_z_position'])
                    
                    # Group by muon_index and draw lines for each group.
                    grouped = current_data_line.groupby('muon_index')
                    for name, group in grouped:
                        # Only plot lines for muons that are relevant for the current frame.
                        if name in relevant_muons:
                            group = group.sort_values(by='detection_time')
                            x = group['detected_x_position'].values
                            y = group['detected_y_position'].values
                            z = group['detected_z_position'].values

                            if len(group)==len(self.rpc_list):
                                c = 'lime'
                            elif len(group) == 1:
                                c = 'red'
                            else:
                                c= 'gold'

                            ax.plot(x, y, z, marker='o', markersize=5, linestyle='-', linewidth=2, label=f'Muon Index {name}', color = c)

            if  frame == number_of_frames-1:
                x_accumulated.clear()
                y_accumulated.clear()
                z_accumulated.clear()
                    # Also clear the scatter plot
                scat.remove()
                # Recreate the scatter plot
                scat = ax.scatter([], [], [])
                return scat,
            else:
                return scat,
    
        # Create the animation
        ani = FuncAnimation(fig, update, frames=number_of_frames, interval=20)

        plt.show()

    def play_video_norm(self,muons,df_selected_muons):
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        df_muons = df_selected_muons

        # Initialize empty arrays to store accumulated positions
        x_accumulated, y_accumulated, z_accumulated = [], [], []
        
        for rpc in self.rpc_list:
            z = rpc.height
            width, length, _ = rpc.dimensions

            vertices = np.array([[0, 0, z],
                                [width, 0, z],
                                [width, length, z],
                                [0, length, z]])
            
            # Define the vertices of the rectangle
            faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
            poly3d = Poly3DCollection(faces, alpha=0.01, edgecolors='r', linewidths=1)
            ax.add_collection3d(poly3d)

        sim_time = self.sim_time_var.get()#Simulation time in seconds

        number_of_frames = int(sim_time*50) #1 Frame is 20ms

        def update(frame, x_accumulated = x_accumulated, y_accumulated = y_accumulated,  z_accumulated = z_accumulated, df_muons = df_muons):
                
            #If you would like to keep the trajectory plotted, then just remove ax.cla().
            ax.cla()

            time = frame*(2e7)#time of simulation in ns

            for rpc in self.rpc_list:
                z = rpc.height
                width, length, _ = rpc.dimensions

                vertices = np.array([[0, 0, z],
                                    [width, 0, z],
                                    [width, length, z],
                                    [0, length, z]])
                
                # Define the vertices of the rectangle
                faces = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
                poly3d = Poly3DCollection(faces, alpha=0.01, edgecolors='r', linewidths=1, facecolors='cyan')
                ax.add_collection3d(poly3d)

            scat = ax.scatter([],[],[])
            ax.annotate(f'Simulation time/s = {frame*(20e-3):.2f}', xy=(0.05, 0.95), xycoords='axes fraction', color='black')

            current_data = df_muons[(time - (2e7) < df_muons['detection_time']) & (df_muons['detection_time'] <= time)]
            muon_data = current_data[current_data['Outcome'] != 'dark']  # Exclude dark counts for muon plotting
            dark_count_data = current_data[current_data['Outcome'] == 'dark']  # Filter for dark counts
            x_current = muon_data['detected_x_position'].values
            y_current = muon_data['detected_y_position'].values
            z_current = muon_data['detected_z_position'].values
            # Assuming 'width' and 'length' are defined elsewhere in your code
            conditionx = x_current <= width
            conditiony = y_current <= length
            conditionx1 = x_current >= 0
            conditionx2 = y_current >= 0

            # Combine the conditions using logical AND to ensure we only keep entries that satisfy both
            combined_condition = conditionx & conditiony & conditionx1 & conditionx2

            # Apply the combined condition to all arrays
            x_filtered = x_current[combined_condition]
            y_filtered = y_current[combined_condition]
            z_filtered = z_current[combined_condition] # Now applying filtering to z_current as well

            # Now, all filtered arrays will have the same dimension
            x_accumulated.extend(x_filtered)
            y_accumulated.extend(y_filtered)
            z_accumulated.extend(z_filtered)
            
            if len(x_accumulated) > 40:
                x_accumulated = x_accumulated[-40:]
            if len(y_accumulated) > 40:
                y_accumulated = y_accumulated[-40:]
            if len(z_accumulated) > 40:
                z_accumulated = z_accumulated[-40:]


            scat._offsets3d = (x_accumulated, y_accumulated, z_accumulated)
            if not dark_count_data.empty:
                dark_x = dark_count_data['detected_x_position'].values
                dark_y = dark_count_data['detected_y_position'].values
                dark_z = dark_count_data['detected_z_position'].values
                dark_time = dark_count_data['detection_time'].values
                ax.scatter(dark_x, dark_y, dark_z, c='black', marker='x', label='Dark Count')  # Plot dark counts distinctly

                # ax.scatter(dark_x, dark_y, dark_z, c='black', marker='x', label='Dark Count')  # Plot dark counts distinctly
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-max(rpc.dimensions[0] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[0] for rpc in self.rpc_list)*1.1)
            ax.set_ylim(-max(rpc.dimensions[1] for rpc in self.rpc_list)*0.1, max(rpc.dimensions[1] for rpc in self.rpc_list)*1.1)
            ax.set_zlim(0, max(rpc.height for rpc in self.rpc_list) + 2)
            
            
            muon_detection_times = df_muons.groupby('muon_index')['detection_time'].agg(['min', 'max']).to_dict('index')
            relevant_muons = [index for index, times in muon_detection_times.items() if times['max'] <= time and times['min']> time-(2e7)]
            if relevant_muons:
                # Filter dataframe for relevant muons.
                current_data_line = df_muons[df_muons['muon_index'].isin(relevant_muons) & ((df_muons['Outcome'] == 'Y') | (df_muons['Outcome'] == 'N'))]
                current_data_line = current_data_line.dropna(subset=['detected_x_position', 'detected_y_position', 'detected_z_position'])
                
                
                # Group by muon_index and draw lines for each group.
                grouped = current_data_line.groupby('muon_index')
                for name, group in grouped:
                    # Only plot lines for muons that are relevant for the current frame.
                    if name in relevant_muons:
                        group = group.sort_values(by='detection_time')
                        x = group['detected_x_position'].values
                        y = group['detected_y_position'].values
                        z = group['detected_z_position'].values

                        if len(group)==len(self.rpc_list):
                            c = 'lime'
                        elif len(group) == 1:
                            c = 'red'
                        else:
                            c= 'gold'
                        
                        ax.plot(x, y, z, marker='o', markersize=5, linestyle='-', linewidth=2, label=f'Muon Index {name}', color = c)


            return ax,
    
        # Create the animation
        ani = FuncAnimation(fig, update, frames=number_of_frames, interval=20)

        plt.show()

    def play_video_norm_old(self,muons,df_selected_muons):

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

#TODO:
        # Generate a decaying particle, some set lifetime.
        # Create charge products, trace paths of products, do animation.
        # Run example for ANUBIS tracking station.
        # plot_detected_muons function should plot muon trajectories of tagged muons.
        # Gaussian voltage distribution, overlap with detector strips. Threshold
        #BUG: Clicking add RPC then exiting that window and attempting to edit current RPC gives nonsenical
        #values in the gas_mixture dictionary
        #BUG: ATM NO MUONS SEEM TO MISS 2nd RPC?????
        #IMPLEMENT NEW CHECK HIT WITH CHECKHIT DARK AND STRIPPED ETC!!!!!!!!!!!!!!!!
        #FIX NORMSCALE ALGORITHM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Important:
#add reconstructino algorithms to measure reconstruction efficiency
#API for external use outside of GUI (Start making measurements)
#Connect with Garfield++ in the linux system (Simulation within gas gaps)
#Make presets for quality of life changes





