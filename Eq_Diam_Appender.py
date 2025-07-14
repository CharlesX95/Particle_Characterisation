# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:00:28 2024

@author: Charlie


This code takes a csv file that contains area measurements (titled 'Area'), and
calculates the equivalent diameter. These are then appended as a new column 
titled 'Eq Diam'
"""
import os
import numpy as np
import pandas as pd

# Define the folder paths
folder = './newPET/'

# Function to calculate Equivalent Diameter
def calculate_eq_diam(area):
    """
    Measures the diameter of a circle of a given area

    Parameters
    area : float
        area of a shape

    Returns
    diameter : float
        diameter of a circle with the equivalent area of the shape
    """
    pi = np.pi
    diameter = 2 * np.sqrt(area / pi)
    return diameter

# Function to process files in a given folder
def process_folder(folder_path):
    """
    Goes through a folder and processes every file ending with .csv
    it will calculate the Eq Diam for each entry and append it to the csv
    if the Eq Diam is already in the file it will skip it

    Parameters
    ----------
    folder_path : string
        Name of the folder
        The folder should be saved in the same path as the project
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            if 'Area' in data.columns:           
                if 'Eq Diam' in data.columns:
                    # Skip files with an Eq Diam column already
                    print(f"Eq Diam already calculated for {filename} - skipping")
                else:
                    # Calculate Equivalent Diameter (Eq Diam) and append it as a new column
                    data['Eq Diam'] = calculate_eq_diam(data['Area'])
                    
                    # Save the updated CSV file (overwrite the original file)
                    data.to_csv(file_path, index=False)
                    print(f"Processed {filename} and added 'Eq Diam' column.")
            else:
                print(f"'Area' column not found in {filename}, skipping.")

# Process both folders
process_folder(folder)

print("All files processed.")
