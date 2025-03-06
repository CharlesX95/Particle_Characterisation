# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:31:52 2024

@author: Charlie
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create the "combined_data" folder if it doesn't exist
output_folder = './combined_data/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 1: Load the data
file_10x = './10x_data/PE2_10x.csv'
file_40x = './40x_data/PE2_40x.csv'

source_10x = os.path.basename(file_10x).split('_')[0]  # Extract the source label ('PS1' in this case)

combined_data_full = pd.DataFrame()

pi = np.pi

data_10x = pd.read_csv(file_10x)
data_40x = pd.read_csv(file_40x)

data_10x['Eq Diam'] = 2 * np.sqrt(data_10x['Area'] / pi)
data_40x['Eq Diam'] = 2 * np.sqrt(data_40x['Area'] / pi)

scale_10x = 1.547
scale_40x = 6.125

min_area_10x_px = 12  # in pixels for 10X
min_area_40x_px = 12  # in pixels for 40X

# Convert minimum area to micrometers²
min_area_10x_um2 = (min_area_10x_px / (scale_10x ** 2))
min_area_40x_um2 = (min_area_40x_px / (scale_40x ** 2))

# Crop datasets to exclude unresolvable particles
data_10x_cropped = data_10x[data_10x['Area'] >= min_area_10x_um2]
data_40x_cropped = data_40x[data_40x['Area'] >= min_area_40x_um2]

# Define the overlapping range
overlap_min = 3  # um
overlap_max = 30  # um

# Step 2: Create histograms for the overlapping region
log_bins = np.logspace(np.log10(overlap_min), np.log10(overlap_max), num=20)

hist_10x_overlap, _ = np.histogram(data_10x_cropped['Eq Diam'], bins=log_bins)
hist_40x_overlap, _ = np.histogram(data_40x_cropped['Eq Diam'], bins=log_bins)

non_zero_indices = (hist_10x_overlap > 0) & (hist_40x_overlap > 0)
filtered_hist_10x_overlap = hist_10x_overlap[non_zero_indices]
filtered_hist_40x_overlap = hist_40x_overlap[non_zero_indices]

#Step 3: Calculate scaling factors using either mean or weighted mean
scaling_factors = filtered_hist_10x_overlap / filtered_hist_40x_overlap
#mean_scaling_factor = np.mean(scaling_factors)
weighted_mean_scaling_factor = np.average(scaling_factors, weights=filtered_hist_40x_overlap)
mean_scaling_factor = weighted_mean_scaling_factor  # or mean_scaling_factor


# Step 4: Create a histogram for the full range of the 40X dataset
full_bins = np.logspace(np.log10(data_40x_cropped['Eq Diam'].min()), 
                         np.log10(data_10x_cropped['Eq Diam'].max()), 
                         num=50)

hist_10x_full, _ = np.histogram(data_10x_cropped['Eq Diam'], bins=full_bins)
hist_40x_full, _ = np.histogram(data_40x_cropped['Eq Diam'], bins=full_bins)
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
plt.title(f"Combined Particle Size Distribution {source_10x}")
plt.legend()
plt.savefig(f"{source_10x} combined_synthesized_particle_size_distribution.png", dpi=300)
plt.show()

# Output synthesized counts for further analysis
synthesized_counts_df = pd.DataFrame({
    'Diameter (um)': full_bins[:-1],
    'Synthesized Count': synthesized_counts_full
})

print(synthesized_counts_df)
print(f"mean scaling factor: {mean_scaling_factor}")
