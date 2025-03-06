# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:00:28 2024

@author: Charlie
"""
import os
import numpy as np
import pandas as pd

# Define the folder paths
folder_10x = './10x_data/'
folder_40x = './40x_data/'

# Function to calculate Equivalent Diameter
def calculate_eq_diam(area):
    pi = np.pi
    return 2 * np.sqrt(area / pi)

# Function to process files in a given folder
def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            if 'Area' in data.columns:
                # Calculate Equivalent Diameter (Eq Diam) and append it as a new column
                
                if 'Eq Diam' in data.columns:
                    print(f"Eq Diam already calculated for {filename} - skipping")
                else:
                    data['Eq Diam'] = calculate_eq_diam(data['Area'])
                    
                    # Save the updated CSV file (overwrite the original file)
                    data.to_csv(file_path, index=False)
                    print(f"Processed {filename} and added 'Eq Diam' column.")
            else:
                print(f"'Area' column not found in {filename}, skipping.")

# Process both folders
process_folder(folder_10x)
process_folder(folder_40x)

print("All files processed.")
