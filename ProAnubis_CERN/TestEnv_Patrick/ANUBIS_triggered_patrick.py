from PIL import Image
import h5py
import anubisPlotUtils as anPlot
import json
import numpy as np
import os
import hist as hi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc.
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import sys
import ANUBIS_triggered_functions as ANT

current_directory=  os.path.dirname(os.getcwd())
file_path = current_directory + "\\PartIIIRPC\\ProAnubis_CERN\\ProAnubisData\\60sRun_24_3_4.h5"

data = ANT.importFromHDF5File(file_path)

chanCounts = [0 for x in range(128)]


