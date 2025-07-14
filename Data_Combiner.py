# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:31:52 2024

@author: Charlie

This code takes two size distribution datasets obtained by different microscopy
magnifications and combines them into a single distribution.
The variables are labelled as 1x and 2x, but any combination should work.
Before combining the data-sets it is crucial to have an accurate measure of the 
px/um calibration of each magnification. This can be obtained by taking images 
of a graticule during the measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pi = np.pi

# Create the "combined_data" folder if it doesn't exist
output_folder = './combined_data/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
"""
    Step 1: Input Microscope Parameters
    Here we input the px/um calibration of the microscope and define our 
    resolution limits
"""

# name the magnifications used
mag_1x = "10X"
mag_2x = "40X"

# px/um measurements for each
scale_1x = 1.547
scale_2x = 6.125

#minimum pixel resolution (should be same for both mags with the same camera)
min_area_1x_px = 12  # in pixels for 10X
min_area_2x_px = 12  # in pixels for 40X

# Convert minimum area to micrometers²
min_area_1x_um2 = (min_area_1x_px / (scale_1x ** 2))
min_area_2x_um2 = (min_area_2x_px / (scale_2x ** 2))

"""
    Step 2: Load the data
    Load both csv files. They should be saved as "SampleName_Magnification.csv"
    i.e., 'Nylon_20X.csv' and they should be saved in the same path as the code
"""
# create a string variable of the data path
file_1x = './10x_data/PE2_10x.csv'
file_2x = './40x_data/PE2_40x.csv'

# Extract the sample name
sample_name = os.path.basename(file_1x).split('_')[0]  

# Create an empty dataframe
combined_data_full = pd.DataFrame()

data_1x = pd.read_csv(file_1x)
data_2x = pd.read_csv(file_2x)

data_1x['Eq Diam'] = 2 * np.sqrt(data_1x['Area'] / pi)
data_2x['Eq Diam'] = 2 * np.sqrt(data_2x['Area'] / pi)

# Crop datasets to exclude unresolvable particles
data_1x_cropped = data_1x[data_1x['Area'] >= min_area_1x_um2]
data_2x_cropped = data_2x[data_2x['Area'] >= min_area_2x_um2]

# Define the overlapping range
overlap_min = 3  # um
overlap_max = 30  # um

# Step 2: Create histograms for the overlapping region
log_bins = np.logspace(np.log10(overlap_min), np.log10(overlap_max), num=20)

hist_10x_overlap, _ = np.histogram(data_1x_cropped['Eq Diam'], bins=log_bins)
hist_40x_overlap, _ = np.histogram(data_2x_cropped['Eq Diam'], bins=log_bins)

non_zero_indices = (hist_10x_overlap > 0) & (hist_40x_overlap > 0)
filtered_hist_10x_overlap = hist_10x_overlap[non_zero_indices]
filtered_hist_40x_overlap = hist_40x_overlap[non_zero_indices]

#Step 3: Calculate scaling factors using either mean or weighted mean
scaling_factors = filtered_hist_10x_overlap / filtered_hist_40x_overlap
#mean_scaling_factor = np.mean(scaling_factors)
weighted_mean_scaling_factor = np.average(scaling_factors, weights=filtered_hist_40x_overlap)
mean_scaling_factor = weighted_mean_scaling_factor  # or mean_scaling_factor


# Step 4: Create a histogram for the full range of the 40X dataset
full_bins = np.logspace(np.log10(data_2x_cropped['Eq Diam'].min()), 
                         np.log10(data_1x_cropped['Eq Diam'].max()), 
                         num=50)

hist_10x_full, _ = np.histogram(data_1x_cropped['Eq Diam'], bins=full_bins)
hist_40x_full, _ = np.histogram(data_2x_cropped['Eq Diam'], bins=full_bins)
synthesized_counts_full = hist_40x_full * mean_scaling_factor


# Optional: Plot the counts for visualization
plt.figure(figsize=(10, 6))
plt.hist(full_bins[:-1], bins=full_bins, weights=synthesized_counts_full, alpha=0.5, label='Synthesized 40X', color='red')
plt.hist(full_bins[:-1], bins=full_bins, weights=hist_10x_full, alpha=0.5, label='10X', color='blue')
plt.hist(full_bins[:-1], bins=full_bins, weights=hist_40x_full, alpha=1, label='Original 40X', color='green')
plt.vlines(overlap_min, 0, 10, color = 'black', label = 'lower overlap limit')
plt.vlines(overlap_max, 0, 10, color = 'black', label = 'upper overlap limit')

plt.xscale('log')
plt.xlim(0.6, 3000)
plt.yscale('log')
plt.xlabel('Equivalent Diameter (um)')
plt.ylabel('Count')
plt.title(f"Combined Particle Size Distribution {sample_name}")
plt.legend()
plt.savefig(f"{sample_name} combined_synthesized_particle_size_distribution.png", dpi=300)
plt.show()

# Output synthesized counts for further analysis
synthesized_counts_df = pd.DataFrame({
    'Diameter (um)': full_bins[:-1],
    'Synthesized Count': synthesized_counts_full
})

print(synthesized_counts_df)
print(f"mean scaling factor: {mean_scaling_factor}")
