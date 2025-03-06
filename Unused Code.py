# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:48:55 2024

@author: Charlie
"""
# Function to fit and plot a log-normal distribution
def fit_log_normal(data):
    shape, loc, scale = lognorm.fit(data, floc=0)  # floc=0 ensures a non-negative distribution
    x = np.linspace(min(data), max(data), 100)
    fitted_pdf = lognorm.pdf(x, shape, loc, scale)
    return x, fitted_pdf, shape, scale

# Define a function that sums two log-normal distributions (for bimodal fit)
def bimodal_log_normal(x, shape1, scale1, shape2, scale2, weight):
    log_norm1 = weight * lognorm.pdf(x, shape1, 0, scale1)
    log_norm2 = (1 - weight) * lognorm.pdf(x, shape2, 0, scale2)
    return log_norm1 + log_norm2

def fit_bimodal_log_normal(bin_edges, counts):
    counts = pd.to_numeric(counts, errors='coerce')
    # Filter out non-positive values
    counts = counts[counts > 0]  
    hist, _ = np.histogram(counts, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Ensure we have the same length for histogram and bin centers
    if len(hist) != len(bin_centers):
        # Use a more consistent number of bins for histogram
        hist, _ = np.histogram(counts, bins=bin_centers, density=True)
        
    # Initial guesses for the parameters
    initial_guess = [1.0, np.mean(counts), 2.0, np.mean(counts) * 1.5, 0.5]

    # Fit the composite log-normal model
    params, _ = curve_fit(bimodal_log_normal, bin_centers, hist, p0=initial_guess)

    return params
    
# Function to fit a Weibull distribution
def fit_weibull(data):
    shape, loc, scale = weibull_min.fit(data, floc=0)
    x = np.linspace(min(data), max(data), 100)
    fitted_pdf = weibull_min.pdf(x, shape, loc, scale)
    return x, fitted_pdf, shape, scale

# Function to fit a mixture of log-normal distributions
def fit_mixture_log_normal(data, n_components=2):
    log_data = np.log(data[data > 0])  # Log-transform the data
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(log_data.reshape(-1, 1))
    means = np.exp(gmm.means_).flatten()  # Convert means back to the original scale
    covariances = np.exp(np.sqrt(gmm.covariances_)).flatten()
    component_assignments = gmm.predict(log_data.reshape(-1, 1))
    return means, covariances, component_assignments, gmm

def fit_histograms_to_log_normal(data_dir, method="log-normal"):
    sample_names = []
    combined_hist_dict = {}

    # Read CSV files and store histograms
    for filename in os.listdir(data_dir):
        if filename.endswith("_combined.csv"):
            sample_name = filename.split("_combined.csv")[0]
            sample_names.append(sample_name)
            
            # Read the bin edges and counts from CSV
            hist_data = pd.read_csv(os.path.join(data_dir, filename), header=None)
            bin_edges = hist_data[0].values  # First column as bin edges
            counts = hist_data[1].values      # Second column as counts
            combined_hist_dict[sample_name] = (bin_edges, counts)
    
    # Fit distribution to each sample based on the method
    for sample_name, hist_data in combined_hist_dict.items():
        bin_edges, counts = hist_data
        
        # Log-Normal Fit
        if method == "log-normal":
            x, fitted_pdf, shape, scale = fit_log_normal(counts)
            plt.figure()
            plt.hist(bin_edges[:-1], bins=bin_edges, weights=counts, alpha=0.6, color='g', density=True)
            plt.plot(x, fitted_pdf, label='Fitted Log-Normal Curve')
            plt.title(f'Log-Normal Fit for {sample_name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            print(f"Sample: {sample_name}")
            print(f"Log-Normal Shape: {shape}")
            print(f"Log-Normal Scale (Geometric mean): {scale}")
            print(f"Log-Normal Mean: {np.exp(scale)}")
            print(f"Log-Normal Standard Deviation: {shape}\n")
        
        # Bimodal Log-Normal Fit
        elif method == "bimodal-log-normal":
            params = fit_bimodal_log_normal(bin_edges, counts)
            print(f"Fitted Bimodal Parameters for {sample_name}: {params}")
            
            # Create fitted curve for bimodal log-normal
            x = np.linspace(min(bin_edges), max(bin_edges), 100)
            fitted_pdf = bimodal_log_normal(x, *params)

            # Plot histogram and fitted bimodal curve
            plt.figure()
            plt.hist(bin_edges[:-1], bins=bin_edges, weights=counts, density=True, alpha=0.6, label=f'Histogram of {sample_name}')
            plt.plot(x, fitted_pdf, label='Fitted Bimodal Log-Normal Curve', color='orange')
            plt.title(f'Bimodal Log-Normal Fit for {sample_name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        
        # Weibull Fit
        elif method == "weibull":
            x, fitted_pdf, shape, scale = fit_weibull(counts)
            plt.figure()
            plt.hist(bin_edges[:-1], bins=bin_edges, weights=counts, density=True, alpha=0.6, color='b', label=f'Histogram of {sample_name}')
            plt.plot(x, fitted_pdf, label='Fitted Weibull Curve')
            plt.title(f'Weibull Fit for {sample_name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            print(f"Sample: {sample_name}")
            print(f"Weibull Shape: {shape}, Scale: {scale}\n")
        
        # Mixture Log-Normal Fit
        elif method == "mixture-log-normal":
            means, covariances, assignments, gmm = fit_mixture_log_normal(counts)
            x = np.linspace(min(counts), max(counts), 100)
            fitted_pdfs = np.exp(gmm.score_samples(np.log(x).reshape(-1, 1)))  # Convert log-likelihood back
            plt.figure()
            plt.hist(bin_edges[:-1], bins=bin_edges, weights=counts, density=True, alpha=0.6, color='r', label=f'Histogram of {sample_name}')
            plt.plot(x, fitted_pdfs, label='Fitted Mixture Log-Normal Curve')
            plt.title(f'Mixture Log-Normal Fit for {sample_name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            print(f"Sample: {sample_name}")
            print(f"Fitted Mixture Log-Normal Means: {means}")
            print(f"Fitted Mixture Log-Normal Covariances: {covariances}")
        else:
            pass


import os

# Create the "combined_data" folder if it doesn't exist
output_folder = './combined_data/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 1: Save the new datasets (10X and synthesized 40X) with the Eq Diam to CSV files

# Add 'Eq Diam' back to the original datasets if it was removed
data_10x_cropped['Eq Diam'] = 2 * np.sqrt(data_10x_cropped['Area'] / pi)
data_40x_cropped['Eq Diam'] = 2 * np.sqrt(data_40x_cropped['Area'] / pi)

# Saving the cropped 10X dataset to a CSV file
data_10x_cropped.to_csv(os.path.join(output_folder, 'PS1_10x_combined.csv'), index=False)

# Saving the synthesized 40X dataset (with scaling applied) to a CSV file
data_40x_cropped['Synthesized Count'] = synthesized_counts_full  # Add the synthesized counts column
data_40x_cropped.to_csv(os.path.join(output_folder, 'PS1_40x_synthesized_combined.csv'), index=False)

# Step 2: Write a log file with details about the overlap and synthesized counts

log_file_path = os.path.join(output_folder, 'log.txt')

with open(log_file_path, 'w') as log_file:
    log_file.write(f"Overlap Range (um): {overlap_min} - {overlap_max}\n")
    log_file.write(f"Mean Scaling Factor (used): {scaling_factor_to_use}\n\n")
    log_file.write("Synthesized Counts for Each Bin (in full range):\n")
    
    # Write the synthesized counts from synthesized_counts_df
    for idx, row in synthesized_counts_df.iterrows():
        log_file.write(f"Diameter: {row['Diameter (um)']:.3f} um, Synthesized Count: {int(row['Synthesized Count'])}\n")

print("Datasets and log file have been saved successfully.")


## Old Overlap Scaling Function
# # Define the overlapping range
# overlap_min = 3 # um
# overlap_max = 12  # um

# # Step 2: Create histograms for the overlapping region
# log_bins = np.logspace(np.log10(overlap_min), np.log10(overlap_max), num=6)

# # Get counts in the overlapping region
# hist_10x_overlap, _ = np.histogram(data_10x_cropped['Eq Diam'], bins=log_bins)
# hist_40x_overlap, _ = np.histogram(data_40x_cropped['Eq Diam'], bins=log_bins)

# # Step 3: Calculate scaling factor based on counts in the overlapping region
# scaling_factors = hist_10x_overlap / hist_40x_overlap
# scaling_factors[np.isinf(scaling_factors)] = 0  # Replace infinity with 0
# scaling_factors = np.nan_to_num(scaling_factors)  # Replace NaNs with 0

# # Average scaling factor for the overlapping bins
# mean_scaling_factor = np.mean(scaling_factors)

# hist_data_10x, bin_edges10 = np.histogram(data_10x_cropped['Eq Diam'], bins = log_bins)
# hist_data_40x, bin_edges40 = np.histogram(data_40x_cropped['Eq Diam'], bins = log_bins)

# peaks10, _ = find_peaks(hist_data_10x)
# peaks40, _ = find_peaks(hist_data_40x)

# plt.plot(bin_edges10[peaks10], hist_data_10x[peaks10], "x", color='black', label='10X Peaks')
# plt.plot(bin_edges40[peaks40], hist_data_40x[peaks40], "o", color='black', label='40X Peaks')

# # Plot KDE for both datasets in the overlap region
# sns.kdeplot(data_10x_cropped['Eq Diam'], label='10X', color='blue', fill=True, alpha=0.3)
# sns.kdeplot(data_40x_cropped['Eq Diam'], label='40X', color='red', fill=True, alpha=0.3)

# # Generate x values for KDE
# x_values = np.linspace(min(min_diameters), max(max_diameters), 1000)

# # Calculate KDE values for both datasets
# kde_values_10x = sns.kdeplot(data_10x_cropped['Eq Diam'], bw_adjust=0.5, grid=False, cut=0).get_lines()[0].get_data()[1]
# kde_values_40x = sns.kdeplot(data_40x_cropped['Eq Diam'], bw_adjust=0.5, grid=False, cut=0).get_lines()[0].get_data()[1]

# # Find overlap region (min of max KDE values)
# overlap_x = np.where((x_values >= min_diameters) & (x_values <= max_diameters))
# overlap_area = simps(np.minimum(kde_values_10x[overlap_x], kde_values_40x[overlap_x]), x_values[overlap_x])

# # Annotate peaks
# for peak in peaks10:
#     plt.annotate(f'{bin_edges10[peak]:.2f}', (bin_edges10[peak], hist_data_10x[peak]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# for peak in peaks40:
#     plt.annotate(f'{bin_edges40[peak]:.2f}', (bin_edges40[peak], hist_data_40x[peak]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)


# Find overlap region
max_size_40x = data_40x_cropped['Eq Diam'].max()
min_size_10x = data_10x_cropped['Eq Diam'].min()

if max_size_40x < min_size_10x:
    print("No overlap region between 10X and 40X datasets.")
else:
    print(f"Overlap Region: {max_size_40x} to {min_size_10x} micrometers")
