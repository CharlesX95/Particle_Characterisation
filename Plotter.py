# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:43:45 2024

@author: Charlie
"""
import os
import seaborn as sns
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import glob
# from scipy.stats import ks_2samp, wasserstein_distance, entropy
# from math import sqrt
# from scipy.stats import lognorm, weibull_min
# from scipy.optimize import curve_fit
# from sklearn.mixture import GaussianMixture

# Define the folder paths
folder_10x = './10x_data/'
folder_40x = './40x_data/'

figure_folder = './figures/'

data_dir = './combined_data/'

plt.rcParams.update({'font.size': 28})


def extract_sample_name(file_name):
    """Extracts sample name from a given file name using regex."""
    # Match sample name assuming it's the first part before '_10x' or '_40x'
    match = re.match(r"(.*)_(10x|40x).*", file_name)
    if match:
        return match.group(1)  # Return the part before _10x or _40x
    else:
        raise ValueError(f"Invalid file name format: {file_name}")

def load_and_filter_data(file, scale, px_res=16):
    """Loads data from a CSV file, applies area filtering based on pixel resolution and scale."""
    data = pd.read_csv(file)
    min_area_um2 = (px_res / (scale ** 2))  # in um2
    data_filtered = data[data['Area'] >= min_area_um2]
    return data_filtered

def load_sample_data(sample_name, px_res=16):
    file_patterns = [
        os.path.join(folder_10x, f"{sample_name}_10x*.csv"),
        os.path.join(folder_40x, f"{sample_name}_40x*.csv")
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    file_10x = None
    file_40x = None
    
    for file in all_files:
        if '_10x' in file:
            file_10x = file
        elif '_40x' in file:
            file_40x = file
    
    if not file_10x:
        raise FileNotFoundError(f"10X file not found for sample: {sample_name}")
    
    if not file_40x:
        raise FileNotFoundError(f"40X file not found for sample: {sample_name}")
    
    scale_10x = 1.547
    scale_40x = 6.125
    
    # Load and filter data using the helper function
    data_10x = load_and_filter_data(file_10x, scale_10x, px_res)
    data_40x = load_and_filter_data(file_40x, scale_40x, px_res)
    
    return data_10x, data_40x

def scatter_plot(sample_name, x_var, y_var, KDEplot=True):
    data_10x, data_40x = load_sample_data(sample_name)
    
    plt.figure(figsize=(8, 7.5))
    plt.scatter(data_10x[x_var], data_10x[y_var], alpha=0.6, label='10X', color='blue')
    plt.scatter(data_40x[x_var], data_40x[y_var], alpha=0.6, label='40X', color='red')
    
    if KDEplot:
        sns.kdeplot(x=data_10x[x_var], y=data_10x[y_var], color = 'black')
    
    plt.xlabel(f"{x_var} ($\mathrm{{\mu m}}$)")
    plt.ylabel(y_var + ' (log-scale)')
    
    # Ensure logarithmic scaling
    plt.yscale('log')
    plt.xscale('log')
    
    plt.xlim(0.5, 3000)
    plt.ylim(1.0, 30)
    
    # Manually set specific ticks
    ax = plt.gca()  # Get current axis
    ax.yaxis.set_major_locator(mticker.FixedLocator([1.0, 3.0, 10, 30]))
    #ax.yaxis.set_minor_locator(mticker.NullLocator())
    
    # Custom formatter for plain (non-scientific) tick labels
    ax.get_yaxis().set_major_formatter(mticker.FixedFormatter(['1', '3', '10', '30']))
    

    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.3)
    

# Function to plot an overlapping histogram with log-log scaling and log bins
def log_histogram_2mag(sample_name, variable, binmin=1, binmax = 30, bin_count=30):
    data_10x, data_40x = load_sample_data(sample_name)

    log_bins = np.logspace(np.log10(binmin), np.log10(binmax), bin_count)
    # Plot histogram with log-log scaling
    plt.figure(figsize=(8, 7.5))
    plt.hist(data_10x[variable], bins=log_bins, alpha=0.6, label='10X', color='blue')
    plt.hist(data_40x[variable], bins=log_bins, alpha=0.8, label='40X', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(binmin, binmax)
    
   
    # # Manually set specific ticks
    # ax = plt.gca()  # Get current axis
    # ax.set_xticks([0.1, 1.0])  # Explicitly set the ticks for the y-axis
    
    # # Custom formatter for plain (non-scientific) tick labels
    # ax.get_xaxis().set_major_formatter(mticker.FixedFormatter(['0.1', '1.0']))
    
    # Manually set specific ticks
    ax = plt.gca()  # Get current axis
    ax.set_xticks([1.0, 3.0, 10, 30])  # Explicitly set the ticks for the y-axis
    
    # Custom formatter for plain (non-scientific) tick labels
    ax.get_xaxis().set_major_formatter(mticker.FixedFormatter(['1', '3', '10', '30']))
    
    

   
    
    plt.xlabel(variable + ' (log-scale)')
    plt.ylabel('Frequency')
    #plt.title(f"{sample_name} : {variable} Unscaled")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=0.3)
    #plt.show()

def data_combiner2(sample_name, variable='Eq Diam', olapmin=5, olapmax=50, olapbins=20, bin_count=50):
    # Load sample data
    data_10x, data_40x = load_sample_data(sample_name)
    
    # Define log bins for the histogram
    log_bins = np.logspace(np.log10(0.6), np.log10(2000), bin_count)
    
    # Overlapping region bins
    overlap_bins = np.logspace(np.log10(olapmin), np.log10(olapmax), num=olapbins)
    hist_10x_overlap, _ = np.histogram(data_10x[variable], bins=overlap_bins)
    hist_40x_overlap, _ = np.histogram(data_40x[variable], bins=overlap_bins)

    # Ignore zero counts when calculating scaling factors
    non_zero_indices = (hist_10x_overlap > 0) & (hist_40x_overlap > 0)
    scaling_factors = hist_10x_overlap[non_zero_indices] / hist_40x_overlap[non_zero_indices]
    mean_scaling_factor = np.average(scaling_factors, weights=hist_40x_overlap[non_zero_indices])

    # Apply scaling factor to the full 40X dataset histogram
    hist_10x_full, _ = np.histogram(data_10x[variable], bins=log_bins)
    hist_40x_full, _ = np.histogram(data_40x[variable], bins=log_bins)
    synthesized_counts_full = hist_40x_full * mean_scaling_factor

    # Combine counts
    combined_counts_full = np.where(
        (hist_10x_full > 0) & (synthesized_counts_full > 0), 
        (hist_10x_full + synthesized_counts_full) / 2, 
        np.where(hist_10x_full > 0, hist_10x_full, synthesized_counts_full)
    )

    # Pre-calculate number fraction
    total_counts = np.sum(combined_counts_full)
    number_fraction = (combined_counts_full / total_counts) * 100

    # Save combined data
    df = pd.DataFrame({'bin_edges': log_bins[:-1], 'counts': combined_counts_full})
    df.to_csv(f'./combined_data/{sample_name}_combined.csv', header=False, index=False)

    # Check if MasterSizer data is available
    master_path = f"./MasterSizer_data/{sample_name}_MasterSizer.csv"
    if os.path.exists(master_path):
        # Load MasterSizer data
        master_data = pd.read_csv(master_path, header=None)
        x_master = master_data[0]
        y_master = master_data[1]

        master_overlay = True
    else:
        print(f"MasterSizer data not found for sample: {sample_name}")
        master_overlay = False

    # Plot combined histogram with pre-calculated number fraction
    plt.figure(figsize=(22.5, 12))

    # Histogram of combined data
    plt.bar(
        log_bins[:-1], number_fraction, 
        width=np.diff(log_bins), align='edge', alpha=0.9, color='red', label='Combined Data'
    )

    # Overlay MasterSizer curve if available
    if master_overlay:
        plt.plot(x_master, y_master, color='blue', linewidth=2, label='MasterSizer')

    # Add overlap region limits
    plt.vlines([olapmin, olapmax], 0, max(number_fraction) * 0.8, color='black', linestyle='dashed', label='Overlap Limits')


    # Plot details
    plt.xlabel('Size (µm) [Log Scale]', fontsize=12)
    plt.ylabel('Number Fraction', fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f'Overlay of {sample_name} Size Distributions', fontsize=14)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()



def data_combiner(sample_name, variable = 'Eq Diam', olapmin=5, olapmax=50, olapbins = 20, bin_count = 50):

    data_10x, data_40x = load_sample_data(sample_name)
    
    # Define log bins for the histogram - must be consistent for multiple samples
    log_bins = np.logspace(np.log10(0.6), np.log10(2000), bin_count)
    
    # Overlapping region bins
    overlap_bins = np.logspace(np.log10(olapmin), np.log10(olapmax), num=olapbins)
    hist_10x_overlap, _ = np.histogram(data_10x[variable], bins=overlap_bins)
    hist_40x_overlap, _ = np.histogram(data_40x[variable], bins=overlap_bins)

    # Ignore zero counts when calculating scaling factors
    non_zero_indices = (hist_10x_overlap > 0) & (hist_40x_overlap > 0)
    scaling_factors = hist_10x_overlap[non_zero_indices] / hist_40x_overlap[non_zero_indices]
    mean_scaling_factor = np.average(scaling_factors, weights=hist_40x_overlap[non_zero_indices])

    # Apply scaling factor to the full 40X dataset histogram
    hist_10x_full, _ = np.histogram(data_10x[variable], bins=log_bins)
    hist_40x_full, _ = np.histogram(data_40x[variable], bins=log_bins)
    synthesized_counts_full = hist_40x_full * mean_scaling_factor

    # Combine counts as per the specified logic
    combined_counts_full = np.where(
        (hist_10x_full > 0) & (synthesized_counts_full > 0),  # Both counts are non-zero
        (hist_10x_full + synthesized_counts_full) / 2,        # Average the counts
        np.where(hist_10x_full > 0,                           # Only 10X count is non-zero
                 hist_10x_full,
                 synthesized_counts_full)                     # Only synthesized count is non-zero
    )
    
    # Pre-calculate number fraction
    total_counts = np.sum(combined_counts_full)
    number_fraction = (hist_40x_full / total_counts) * 100

    
    df = pd.DataFrame({'bin_edges': log_bins[:-1], 'counts': combined_counts_full})    
    df.to_csv(f'./combined_data/{sample_name}_combined2.csv', header=False, index = False)
    
    
    master_path = f"./MasterSizer_data/{sample_name}_MasterSizer.csv"
    if os.path.exists(master_path):
        # Load MasterSizer data
        master_data = pd.read_csv(master_path, header=None)
        x_master = master_data[0]
        y_master = master_data[1]

        master_overlay = True
    else:
        print(f"MasterSizer data not found for sample: {sample_name}")
        master_overlay = False

    
    #plotting to visualise
    plt.figure(figsize=(22.5, 12))
    
    
05    
    #plt.hist(data_10x[variable], bins=log_bins, alpha=1, color='blue')
    # plt.hist(data_40x[variable], bins=log_bins, alpha=0.1, color='green')
    # plt.hist(log_bins[:-1], bins=log_bins, weights=synthesized_counts_full, alpha=0.1, color='green')
    # plt.hist(log_bins[:-1], bins=log_bins, weights=combined_counts_full, alpha=0.1,  color='red')
    
    # plt.step(log_bins[:-1], np.histogram(data_10x[variable], bins=log_bins)[0], where='post', linestyle='solid', color='blue', label='10X')
    # plt.step(log_bins[:-1], np.histogram(data_40x[variable], bins=log_bins)[0], where='post', linestyle='solid', color='green', label='40X')
    # plt.step(log_bins[:-1], synthesized_counts_full, where='post', linestyle='dotted', color='green', label='Synthesized 40X')
    # plt.step(log_bins[:-1], combined_counts_full, where='post', linestyle='solid', linewidth = 1.5, color='red', label='Combined')
    
    # #plot details
    # plt.vlines([olapmin, olapmax], 0, 10, color='black', label='Overlap Limits')



    
    plt.hist(log_bins[:-1], bins=log_bins, weights=number_fraction, alpha=0.7,  color='blue', density = False, label = '40X Microscopy')

    if master_overlay:
        plt.plot(x_master, y_master, color='black', linewidth=5, label='MasterSizer')
    averages = calculate_weighted_averages(df)
    
    # # Annotate each weighted average on the plot
    # y_position = max(combined_counts_full) * 0.8  # Starting y position for text, adjust as needed
    # for label, value in averages.items():
    #     plt.vlines(value, ymin=0, ymax=1, color='black', linestyle='dashed', linewidth=5, label = (f'{label} = {value:.1f}')+('$\mathrm{\mu m}$'))
    #     # plt.text(value, 10, f"{label}: {value:.2f}",
    #     #      verticalalignment='top', horizontalalignment='center', color='black')

    #     #plt.text(value, 10.5, f"N-{value:.2f}")
    # print(averages)
        
    


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Eq. Diam. ($\mathrm{\mu m}$) (log-scale)')
    plt.ylabel('Number Fraction (%) (log-scale)')
    plt.ylim(0.01, 100)
    plt.xlim(0.4, 2000)
    plt.title(f"{sample_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # if master_overlay:    
    #      plt.show()
    
def calculate_weighted_averages(df):
    # Extract bin edges and counts from the DataFrame
    bin_edges = df['bin_edges'].values
    counts = df['counts'].values
    
    # Calculate bin midpoints from bin edges
    #bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_midpoints = bin_edges
    
    
    # Calculate equivalent diameters for length, area, and volume calculations
    diameter_1 = bin_midpoints
    diameter_2 = bin_midpoints ** 2
    diameter_3 = bin_midpoints ** 3
    diameter_4 = bin_midpoints ** 4
    
    # Calculate weighted averages
    number_weighted_avg = round((np.sum(diameter_1 * counts) / np.sum(counts)), 2)
    length_weighted_avg = round((np.sum(diameter_2 * counts) / np.sum(diameter_1 * counts)), 2)
    area_weighted_avg = round((np.sum(diameter_3 * counts) / np.sum(diameter_2 * counts)), 2)
    volume_weighted_avg = round((np.sum(diameter_4 * counts) / np.sum(diameter_3 * counts)), 2)
    
    
    
    # Return results as a dictionary
    return {
        "No. Avg.": number_weighted_avg,
        "Len. Avg.": length_weighted_avg,
        "Area Avg.": area_weighted_avg,
        "Vol. Avg.": volume_weighted_avg
    }


def log_histogram_plot(sample_name, mag = '10x', variable='Eq Diam', binmin=0.7, binmax = 1000, bin_count=60):
    
    data_10x, data_40x = load_sample_data(sample_name)
    log_bins = np.logspace(np.log10(binmin), np.log10(binmax), bin_count)
    
    if mag == '10x':
        data = data_10x
    elif mag == '40x':
        data = data_40x
    
    # Plot histogram with log-log scaling
    plt.figure(figsize=(10, 6))
    plt.hist(data[variable], bins=log_bins, alpha=0.4, label='10X', color='blue')

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(f"{variable}")
    plt.ylabel('Frequency')
    plt.ylim(1, 1.2)
    plt.xlim(binmin, binmax)
    plt.title(f"{sample_name} : {variable}")
    plt.legend()
    plt.grid(True)
    plt.show()

  
def normalise_histogram(hist):
    total = np.sum(hist)
    return hist / total if total > 0 else hist

def jsd(hist1, hist2, eps=1e-10):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    
    # M is the average distribution
    M = (hist1_norm + hist2_norm) / 2
    
    # Add epsilon to avoid division by zero or log(0)
    kl1 = np.sum(hist1_norm * np.log(hist1_norm / (M + eps) + eps))
    kl2 = np.sum(hist2_norm * np.log(hist2_norm / (M + eps) + eps))
    
    # Calculate JSD
    jsd_value = 0.5 * (kl1 + kl2)
    
    return jsd_value

def kl_div(hist1, hist2):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    return np.sum(hist1_norm * np.log10((hist1_norm + 1e-10) / (hist2_norm + 1e-10)))
    
def wasserstein(hist1, hist2):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    return wasserstein_distance(hist1_norm, hist2_norm)

def hellinger(hist1, hist2):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1_norm) - np.sqrt(hist2_norm))**2))
                
def ks_test(hist1, hist2):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    return ks_2samp(hist1_norm, hist2_norm).statistic

def bhatt(hist1, hist2):
    hist1_norm = normalise_histogram(hist1)
    hist2_norm = normalise_histogram(hist2)
    return -np.log(np.sum(np.sqrt(hist1_norm*hist2_norm)))

def compare_samples(method, data_dir=data_dir):
    sample_names = []
    combined_counts_dict = {}
    
    # Read CSV files and store histograms
    for filename in os.listdir(data_dir):
        if filename.endswith("_combined.csv"):
            sample_name = filename.split("_combined.csv")[0]
            sample_names.append(sample_name)
            
            # Read the histogram data from CSV
            hist_data = pd.read_csv(os.path.join(data_dir, filename), header=None)
            combined_counts_dict[sample_name] = hist_data[1].values  # gets only the counts
            print(f"{sample_name} histo data : {hist_data}")
    
            #print(f"combined counts : {combined_counts_dict}")
     
    # Initialize match scores dictionary
    match_scores = {}
    for i, sample_a in enumerate(sample_names):
        match_scores[sample_a] = {}
        for j, sample_b in enumerate(sample_names):
            if sample_a != sample_b:
                if method == "JSD":
                    score = jsd(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                elif method == "KL":
                    score = kl_div(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                elif method == "Wasserstein":
                    score = wasserstein(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                elif method == "KS":
                    score = ks_test(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                elif method == "Hellinger":
                    score = hellinger(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                elif method == "Bhatt":
                    score = bhatt(combined_counts_dict[sample_a], combined_counts_dict[sample_b])
                
                match_scores[sample_a][sample_b] = score
            else:
                match_scores[sample_a][sample_b] = 0.0  # Self-comparison gives 0

    # If no reference or not in sample_names, return original sample_names
    return match_scores, sample_names  # Return match_scores and sample_names

def plot_heatmap(method, vmin = 0, vmax = 1, reference=None):
    
    match_scores, sample_names = compare_samples(method)  # Get match_scores and sample_names
    
    # If a reference sample is provided and exists in the sample names, sort by similarity to the reference
    if reference and reference in sample_names:
        reference_scores = match_scores[reference]
        # Sort samples by their similarity to the reference (ascending order of similarity score)
        sorted_samples = sorted(sample_names, key=lambda x: reference_scores[x])
    else:
        sorted_samples = sample_names  # No reference, keep the original order

    # Create a DataFrame with the similarity scores using sorted samples
    match_df = pd.DataFrame(index=sorted_samples, columns=sorted_samples)

    for sample_a in sorted_samples:
        for sample_b in sorted_samples:
            match_df.loc[sample_a, sample_b] = match_scores[sample_a][sample_b]

    # Create a figure and a heatmap with a custom colormap
    plt.figure(figsize=(14, 12))
    
    # Use seaborn's heatmap, with 'RdYlGn' colormap (red to green)
    sns.heatmap(match_df.astype(float), annot=True, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    
    # Set the axis labels and title
    plt.title(f"{method} Method Similarity between Samples")
    # plt.xlabel("Samples")
    # plt.ylabel("Samples")
    plt.gca().xaxis.tick_top()
    
    # Display the heatmap
    plt.show()
    
# Fit a log-normal distribution to midpoints with weights (counts)
def fit_lognormal(bin_midpoints, pdf):
    shape, loc, scale = lognorm.fit(bin_midpoints, floc=0, scale=1, fscale=np.average(bin_midpoints, weights=pdf))
    
    x = np.logspace(np.log10(bin_midpoints.min()), np.log10(bin_midpoints.max()), 500)
    pdf_lognorm_fitted = lognorm.pdf(x, shape, loc, scale)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(bin_midpoints, pdf, alpha=1, label='Measured PDF', color='skyblue')
    plt.plot(x, pdf_lognorm_fitted, 'r-', lw=2, label='Fitted Log-Normal PDF')
    plt.xlabel('Particle Size (units)')
    plt.ylabel('Probability Density')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Fitted Log-Normal PDF')
    plt.legend()
    plt.show()
    return shape, loc, scale

# Fit GMM model using midpoints and counts as weights
def fit_gmm(bin_midpoints, pdf, n_components=2):
    bin_midpoints = bin_midpoints.reshape(-1, 1)  # Reshape for GMM
    
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(bin_midpoints, pdf)
    
    x = np.logspace(np.log10(bin_midpoints.min()), np.log10(bin_midpoints.max()), 500).reshape(-1, 1)
    pdf_fitted = np.exp(gmm.score_samples(x))
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(bin_midpoints, pdf, label='Measured PDF', color='skyblue')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'GMM Fit ({n_components} components)')
    plt.xlabel('Particle Size (units)')
    plt.ylabel('Probability Density')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('GMM Fitted PDF')
    plt.legend()
    plt.show()
    
    return gmm

# Fit a bimodal log-normal distribution to the midpoints and pdf
def fit_bimodal_lognormal(bin_midpoints, pdf):
    positive_midpoints = bin_midpoints[bin_midpoints > 0]
    positive_pdf = pdf[:len(positive_midpoints)]
    
    split_index = int(len(positive_midpoints) * 0.6)
   
    midpoints1 = positive_midpoints[:split_index]
    midpoints2 = positive_midpoints[split_index:]
    pdf1 = positive_pdf[:split_index]
    pdf2 = positive_pdf[split_index:]
    
    shape1, loc1, scale1 = lognorm.fit(midpoints1, floc=0, scale=1, fscale=np.average(midpoints1, weights=pdf1))
    shape2, loc2, scale2 = lognorm.fit(midpoints2, floc=0, scale=1, fscale=np.average(midpoints2, weights=pdf2))
    
    x = np.logspace(np.log10(bin_midpoints.min()), np.log10(bin_midpoints.max()), 500)
    pdf_fitted1 = lognorm.pdf(x, shape1, loc1, scale1)
    pdf_fitted2 = lognorm.pdf(x, shape2, loc2, scale2)
    
    pdf_fitted_bimodal = (pdf_fitted1 + pdf_fitted2) / 2
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(positive_midpoints, positive_pdf, label='Measured PDF', color='skyblue')
    plt.plot(x, pdf_fitted_bimodal, 'r-', lw=2, label='Bimodal Log-Normal Fit')
    plt.xlabel('Particle Size (units)')
    plt.ylabel('Probability Density')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Bimodal Log-Normal Fitted PDF')
    plt.legend()
    plt.show()
    
    return (shape1, loc1, scale1), (shape2, loc2, scale2)

# Main function to choose fitting method
def fit_pdf(data_dir=data_dir, method='log-normal'):
    for filename in os.listdir(data_dir):
        if filename.endswith("_combined.csv"):
            sample_name = filename.split("_combined.csv")[0]
            
            data = pd.read_csv(os.path.join(data_dir, filename), header=None).values
            bin_edges = data[:, 0]
            counts = data[:, 1]
            
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            total_counts = np.sum(counts)
            pdf = counts / total_counts
            
            pdf = pdf[:len(bin_midpoints)]  # Ensure matching lengths
            
            if method == 'gmm':
                gmm = fit_gmm(bin_midpoints, pdf)
                print(f"{sample_name} GMM: {gmm}")
            elif method == 'log-normal':
                shape, loc, scale = fit_lognormal(bin_midpoints, pdf)
                print(f"{sample_name} Log-Normal: {shape, loc, scale}")
            elif method == 'bimodal':
                (shape1, loc1, scale1), (shape2, loc2, scale2) = fit_bimodal_lognormal(bin_midpoints, pdf)
                print (f"{sample_name} Bimodal: {(shape1, loc1, scale1), (shape2, loc2, scale2)}")
            else:
                print("Invalid method")

def save_current_fig(filename, plot_type, variable = None, x_var = None, y_var = None, folder=figure_folder, dpi=300):
    if plot_type == 'scatter' and x_var and y_var:
        subfolder = f"{folder}/scatter_{x_var}_vs_{y_var}_KDE_new"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    elif plot_type == 'scale_log_histogram' or 'scale_log_histogram_2':
        subfolder = f"{folder}/log_histogram_{variable}_40X_ONLY"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    else:
        subfolder = f"{folder}/{plot_type}_{variable}_new2222"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
    safe_filename = filename.replace('.', '').replace('/', '_').replace('\\', '_')
    file_path = os.path.join(subfolder, safe_filename)
    plt.savefig(file_path, dpi=dpi)
    plt.close()
    print(f"Figure saved as {file_path}")

def plot_and_save_samples(variables, plot_types, isSave = False ):
    # Get all unique sample names
    all_files = glob.glob(os.path.join(folder_10x, "*.csv")) + glob.glob(os.path.join(folder_40x, "*.csv"))
    sample_names = set([extract_sample_name(os.path.basename(f)) for f in all_files if extract_sample_name(f)])
    
    # Create output folder if not exists
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
    if isSave == False:
        for sample_name in sample_names:
            for variable, plot_type in zip(variables, plot_types):
                if plot_type == 'scale_log_histogram':
                    log_histogram_2mag(sample_name, variable)
                elif plot_type == 'log_histogram':
                    log_histogram_plot(sample_name, variable)
                elif plot_type == 'scatter':
                    x_var, y_var = variable
                    scatter_plot(sample_name, x_var, y_var, KDEplot = False)
                elif plot_type == 'data_combiner':
                        print(sample_name)
                        data_combiner(sample_name)
                        

    
    if isSave == True:
        for sample_name in sample_names:
            data_10x, data_40x = load_sample_data(sample_name)
            
            for variable, plot_type in zip(variables, plot_types):
                if plot_type == 'scale_log_histogram':
                    log_histogram_2mag(sample_name, variable)
                    save_current_fig(f"{sample_name}_{variable}_{plot_type}_comb_nonlog", plot_type, variable=variable)
                elif plot_type == 'log_histogram':
                    log_histogram_plot(sample_name, data_10x, data_40x, variable)
                    save_current_fig(f"{sample_name}_{variable}_{plot_type}", plot_type, variable=variable)
                elif plot_type == 'scatter':
                    x_var, y_var = variable  # unpack the tuple of variables (x, y)
                    scatter_plot(sample_name, x_var, y_var)
                    save_current_fig(f"{sample_name}_{variable}_{plot_type}_KDE__", plot_type, x_var = x_var, y_var = y_var)
                elif plot_type == 'data_combiner':
                    data_combiner(sample_name)
                    save_current_fig(f"{sample_name}_{variable}_40X_ONLY", plot_type, variable=variable)
                    print("data saved")

# Example usage
variables = [('Eq Diam')] # Example: scatter and log histogram
plot_types = ['data_combiner']

plot_and_save_samples(variables, plot_types, isSave = True)

#plot_heatmap("Hellinger", vmax = 0.4, reference= 'Nylon 6')

#fit_histograms_to_log_normal(data_dir=data_dir, method = 'bimodal-log-normal')
#fit_pdf(method = 'bimodal')