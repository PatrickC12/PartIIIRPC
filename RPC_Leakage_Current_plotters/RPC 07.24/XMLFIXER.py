import numpy as np
import xml.etree.ElementTree as ET
import os
import pandas as pd 
import re

current_directory = os.getcwd()

file_path= current_directory + "\\RPC_Leakage_Current_plotters\\RPC 07.24\\withoutBooks.xml"


# Read the XML file
with open(file_path) as file:
    xml_content = file.read()

# Replace 'DC/AC' with 'DC_AC' using regular expressions
modified_content = re.sub(r'DC/AC', 'DC_AC', xml_content)

# Write the modified content back to the file
with open(current_directory+'withoutBooks_modified.xml', 'w') as file:
    file.write(modified_content)

print("Replacement complete. Modified XML file saved.")