# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:49:37 2024

@author: Charlie
"""

import os
import pandas as pd

# Folder paths
combined_data_folder = './combined_data'
volume_data_folder = './volume_data'

# Ensure the volume_data folder exists
os.makedirs(volume_data_folder, exist_ok=True)

# Function to calculate the volume fraction
def calculate_volume_fraction(eq_diam, count):
    # Assuming spherical particles, the volume of each particle is proportional to (Eq Diam)^3.
    # Volume fraction is the sum of individual particle volumes divided by the total volume.
    # Eq Diam is in micrometers, so we cube it (and consider scaling factors if needed).
    particle_volume = (eq_diam ** 3)
    total_volume = sum(particle_volume * count)
    return (particle_volume * count) / total_volume

# Loop through each CSV file in the combined_data folder
for filename in os.listdir(combined_data_folder):
    if filename.endswith('_combined.csv'):
        # Read the CSV file
        filepath = os.path.join(combined_data_folder, filename)
        data = pd.read_csv(filepath, header=None, names=['Eq Diam', 'Count'])
        
        # Calculate the volume fraction
        data['Volume Fraction'] = calculate_volume_fraction(data['Eq Diam'], data['Count'])
        
        # Save the result as a new CSV file
        output_filepath = os.path.join(volume_data_folder, filename.replace('_combined.csv', '_volume_fraction.csv'))
        data.to_csv(output_filepath, index=False)
        print(f"Volume fraction data saved for {filename}")
