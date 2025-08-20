import os
#import seaborn as sns
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import glob

"""
User Inputs
"""
# Choose which magnifications to compare or combine
# data_combiner will scale mag_2 data using mag_1 data
# log_histogram will just use mag_1
mag_1 = '10x'
mag_2 = '40x'

# Pixel limit of the optical set-up
# 16 pixels is a reasonable minimum
px_res = 25

# Type of Plot: log_histogram, log_histogram_2mag, scatter, data_combiner
plot_types = ['data_combiner']

# Variable or variables to plot
# Use two variables for scatter plot: e.g., [('Eq Diam', 'AR')]
# data_combiner is only for a size paramater, e.g., Eq Diam or Feret
variables = [('Eq Diam')]

# Bin limits for histograms :: lower, upper, and number of bins
binmin = 0.5
binmax = 3000
bin_count = 30

# For the data_combiner, choose a range which both magnifications accurately measure  
olapmin = 5
olapmax = 100

# X-limits :: For histograms it's wise to set this to the binmin and binmax
x_lower = binmin-(0.1*binmin)
x_upper = binmax+(0.1*binmax)

# Y-limits
y_lower = 1
y_upper = 30000

# Do you want to save or view the figs 
save_file = False

# Source folders for the data :: MUST EXIST
folder_1x = f'./{mag_1}_data/'
folder_2x = f'./{mag_2}_data/'

# Output folder for any saved figs :: Doesn't have to exist yet
figure_folder = './figures/pres_figures'

# Output folder for combined data sets :: Note this will only be a list of bins and heights
data_dir = './combined_data/'

"""
Cosmetic Changes
"""
fontsize = 24
font ='Arial'
color1 = 'blue'
color2 = 'red'
color3 = 'red'
figwidth = 15
figheight = 8
show_overlaps = False
weighted_averages = True
normalised = False
transparency = 0.5

if normalised:
    y_lower = 0.001
    y_upper= 1.1
"""
Ignore Beyond Here!
"""

viewFig = not save_file

plt.rcParams.update({
    'font.family': font,
    'font.size': fontsize,
    'axes.formatter.min_exponent': 3
    })

# this magnification to px/um mapping is for the DFK microscope
# add or edit your own if needs be 
mag_to_scale = {
    '4x': 0.773,
    '10x': 1.936,
    '20x': 3.872,
    '40x': 7.744,
    '50x': 9.68,
    '63x': 12.206
}

scale_1x = mag_to_scale[mag_1]
scale_2x = mag_to_scale[mag_2]

def scatter_plot(sample_name, x_var, y_var, KDEplot=False):
    data_1x, data_2x = load_sample_data(sample_name)
    
    plt.figure(figsize=(figwidth, figheight))
    plt.scatter(data_1x[x_var], data_1x[y_var], alpha=0.6, label=mag_1, color=color1)
    plt.scatter(data_2x[x_var], data_2x[y_var], alpha=0.6, label=mag_2, color=color2)
    
    max_x = np.max(data_1x[x_var])
    print(f"Max {x_var} = {max_x}")
    # if KDEplot:
    #     sns.kdeplot(x=data_1x[x_var], y=data_1x[y_var], color = 'black')
    
    plt.xlabel(f"{x_var} ($\mathrm{{\mu m}}$)")
    plt.ylabel(y_var + ' (log-scale)')
    
    # Ensure logarithmic scaling
    plt.yscale('log')
    plt.xscale('log')
    
    plt.xlim(x_lower, x_upper)
    plt.ylim(y_lower, y_upper)
    
    # # Manually set specific ticks
    # ax = plt.gca()  # Get current axis
    # ax.yaxis.set_major_locator(mticker.FixedLocator([1.0, 3.0, 10, 30]))
    # #ax.yaxis.set_minor_locator(mticker.NullLocator())
    
    # # Custom formatter for plain (non-scientific) tick labels
    # ax.get_yaxis().set_major_formatter(mticker.FixedFormatter(['1', '3', '10', '30']))
    
    plt.title(f"{sample_name} - {x_var} vs. {y_var}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.3)
    plt.show() if viewFig else None
    
def log_histogram_1mag(sample_name, variable='Eq Diam', binmin=binmin, binmax=binmax, bin_count=bin_count):
    data_1x, _ = load_sample_data(sample_name)
    
    # Create logarithmic bins
    log_bins = np.logspace(np.log10(binmin), np.log10(binmax), bin_count)
    
    # Compute histogram manually
    # counts, bin_edges = np.histogram(data_1x[variable], bins=log_bins)
    # total_counts = np.sum(counts)
    # number_fraction = (counts / total_counts) * 100  # in percent

    # bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean for log bins
    
    plt.figure(figsize=(figwidth, figheight))

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(x_lower, x_upper)
    plt.ylim(y_lower, y_upper)
    
    plt.hist(data_1x[variable], 
             bins=log_bins,
             weights=np.ones_like(data_1x[variable]) * 100 / len(data_1x[variable]),  # normalize to percent
             histtype='stepfilled',
             alpha=0.8,
             label=mag_1,
             color=color1,
             edgecolor='black'
             )
    
    plt.xlabel(variable + ' (log-scale)')
    plt.ylabel('Number Fraction (%) (log-scale)')
    plt.title(f"{sample_name} - {variable} Distribution")
    plt.grid(True)
    plt.legend()
    plt.show() if viewFig else None
    
# Function to plot an overlapping histogram with log-log scaling and log bins
def log_histogram_2mag(sample_name, variable, binmin=binmin, binmax=binmax, bin_count=bin_count):
    data_1x, data_2x = load_sample_data(sample_name)

    log_bins = np.logspace(np.log10(binmin), np.log10(binmax), bin_count)

    # Plot filled bar histograms
    plt.figure(figsize=(figwidth, figheight))
    
    plt.hist(data_1x[variable], 
             bins=log_bins,
             weights=np.ones_like(data_1x[variable]) * 100 / len(data_1x[variable]),  # normalize to percent
             histtype='stepfilled',
             alpha=0.8,
             label=mag_1,
             color=color1,
             edgecolor='black'
             )

    plt.hist(data_2x[variable], 
             bins=log_bins,
             weights=np.ones_like(data_2x[variable]) * 100 / len(data_2x[variable]),  # normalize to percent
             histtype='stepfilled',
             alpha=0.8,
             label=mag_2,
             color=color2,
             edgecolor='black'
             )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(x_lower, x_upper)
    plt.ylim(y_lower, y_upper)

    plt.xlabel(variable + ' (log-scale)')
    plt.ylabel('Number Fraction (%) (log-scale)')
    plt.title(f"{sample_name} - {variable} Distribution")
    plt.grid(True)
    plt.legend()
    plt.show() if viewFig else None



def data_combiner(sample_name, variable = 'Eq Diam', olapmin=olapmin, olapmax=olapmax, olapbins = 20, bin_count = bin_count):

    data_1x, data_2x = load_sample_data(sample_name)
    
    # Define log bins for the histogram - must be consistent for multiple samples
    log_bins = np.logspace(np.log10(0.6), np.log10(2000), bin_count)
    
    binmin = np.min(log_bins)
    binmax = np.max(log_bins)
    
    # Overlapping region bins
    overlap_bins = np.logspace(np.log10(olapmin), np.log10(olapmax), num=olapbins)
    hist_1x_overlap, _ = np.histogram(data_1x[variable], bins=overlap_bins)
    hist_2x_overlap, _ = np.histogram(data_2x[variable], bins=overlap_bins)

    # Ignore zero counts when calculating scaling factors
    non_zero_indices = (hist_1x_overlap > 0) & (hist_2x_overlap > 0)
    scaling_factors = hist_1x_overlap[non_zero_indices] / hist_2x_overlap[non_zero_indices]
    mean_scaling_factor = np.average(scaling_factors, weights=hist_2x_overlap[non_zero_indices])

    # Apply scaling factor to the full 2x dataset histogram
    hist_1x_full, _ = np.histogram(data_1x[variable], bins=log_bins)
    hist_2x_full, _ = np.histogram(data_2x[variable], bins=log_bins)
    synthesized_counts_full = hist_2x_full * mean_scaling_factor

    # Combine counts as per the specified logic
    combined_counts_full = np.where(
        (hist_1x_full > 0) & (synthesized_counts_full > 0),  # Both counts are non-zero
        (hist_1x_full + synthesized_counts_full) / 2,        # Average the counts
        np.where(hist_1x_full > 0,                           # Only 1x count is non-zero
                 hist_1x_full,
                 synthesized_counts_full)                     # Only synthesized count is non-zero
    )
    
    # Pre-calculate number fraction
    total_counts = np.sum(combined_counts_full)
    number_fraction = (hist_2x_full / total_counts) * 100
    
    df = pd.DataFrame({'bin_edges': log_bins[:-1], 'counts': combined_counts_full})    
    df.to_csv(f'./combined_data/{sample_name}_combined2.csv', header=False, index = False)
    
    averages = calculate_weighted_averages(df)
    
    
    #plotting to visualise
    plt.figure(figsize=(figwidth, figheight))
    
    # Show the overlap limits 
    # comment/uncomment
    if show_overlaps:
        plt.vlines([olapmin, olapmax], 0, 100, color='black', linewidth = 4, label='Overlap Limits')
    
    if normalised:
        # Normalised, combined distribution
        plt.hist(log_bins[:-1], bins=log_bins, weights=number_fraction, alpha=0.7,  color=color3, density = False, label = f'Combined {mag_1}, {mag_2}')
    else:
        # Show the full datasets and their manipulations
        # comment/uncomment each row that you want or not

        #plt.hist(data_1x[variable], bins=log_bins, alpha=transparency, color=color1, edgecolor=color1, label = f'{mag_1}')
        
        #plt.hist(data_2x[variable], bins=log_bins, alpha=transparency, color=color2, edgecolor=color2, label = f'{mag_2}')
        
        #plt.hist(log_bins[:-1], bins=log_bins, weights=synthesized_counts_full, alpha=transparency, color=color2, label = f'Rescaled {mag_2}')
        
        plt.hist(log_bins[:-1], bins=log_bins, weights=combined_counts_full, alpha=1.0,  color=color3, edgecolor = 'black', label = 'Combined Dataset')
    

    # Annotate each weighted average on the plot
    if weighted_averages:
        for label, value in averages.items():
            plt.vlines(value, ymin=0, ymax=1, color='black', linestyle='dashed', linewidth=2, label = (f'{label} = {value:.1f}')+(' $\mathrm{\mu m}$'))
    print(averages)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Eq. Diam. ($\mathrm{\mu m}$) (log-scale)')
    plt.ylabel('Count (log-scale)')
    plt.ylim(y_lower, y_upper)
    plt.xlim(x_lower, x_upper)
    plt.title(f"{sample_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() if viewFig else None
    
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
  
def normalise_histogram(hist):
    total = np.sum(hist)
    return hist / total if total > 0 else hist
 
def extract_sample_name(file_name):
    """Extracts sample name from a given file name using regex."""
    pattern = rf"(.*)_({mag_1}|{mag_2}).*"
    match = re.match(pattern, file_name)
    if match:
        return match.group(1)
    # else:
    #     raise ValueError(f"Invalid file name format: {file_name}")

def load_and_filter_data(file, scale, px_res=16):
    """Loads data from a CSV file, applies area filtering based on pixel resolution and scale."""
    data = pd.read_csv(file)
    min_area_um2 = (px_res / (scale ** 2))  # in um2
    data_filtered = data[data['Area'] >= min_area_um2]
    return data_filtered

def load_sample_data(sample_name, px_res=px_res):
    file_patterns = [
        os.path.join(folder_1x, f"{sample_name}_{mag_1}.csv"),
        os.path.join(folder_2x, f"{sample_name}_{mag_2}.csv")
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    file_1x = None
    file_2x = None
    
    for file in all_files:
        if f'_{mag_1}.csv' in file:
            file_1x = file
        elif f'_{mag_2}.csv' in file:
            file_2x = file
    
    if not file_1x:
        raise FileNotFoundError(f"{mag_1} file not found for sample: {sample_name}")
    
    if not file_2x:
        raise FileNotFoundError(f"{mag_2} file not found for sample: {sample_name}")
    
    # Load and filter data using the helper function
    data_1x = load_and_filter_data(file_1x, scale_1x, px_res)
    data_2x = load_and_filter_data(file_2x, scale_2x, px_res)
    
    return data_1x, data_2x   
 
def save_current_fig(filename, plot_type, variable = None, x_var = None, y_var = None, folder=figure_folder, dpi=300):
    if plot_type == 'scatter' and x_var and y_var:
        subfolder = f"{folder}/scatter_{x_var}_vs_{y_var}_KDE_new"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    elif plot_type == 'scale_log_histogram' or 'scale_log_histogram_2':
        subfolder = f"{folder}/log_histogram_{variable}"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    else:
        subfolder = f"{folder}/{plot_type}_{variable}"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
    safe_filename = filename.replace('.', '').replace('/', '_').replace('\\', '_')
    file_path = os.path.join(subfolder, safe_filename)
    plt.savefig(file_path, dpi=dpi)
    plt.close()
    print(f"Figure saved as {file_path}")

def plot_and_save_samples(variables, plot_types, isSave=False):
    # Get all unique sample names
    all_files = glob.glob(os.path.join(folder_1x, "*.csv")) + glob.glob(os.path.join(folder_2x, "*.csv"))
    sample_names = {extract_sample_name(os.path.basename(f)) for f in all_files if extract_sample_name(f)}
    
    # Create output folder if needed
    os.makedirs(figure_folder, exist_ok=True)

    for sample_name in sample_names:
        if isSave:
            data_1x, data_2x = load_sample_data(sample_name)
        for variable, plot_type in zip(variables, plot_types):
            if plot_type == 'log_histogram_2mag':
                log_histogram_2mag(sample_name, variable)
                if isSave:
                    save_current_fig(f"{sample_name}_{plot_type}_{variable}", plot_type, variable=variable)
            elif plot_type == 'log_histogram':
                if isSave:
                    log_histogram_1mag(sample_name, variable)
                    save_current_fig(f"{sample_name}_{plot_type}_{variable}", plot_type, variable=variable)
                else:
                    log_histogram_1mag(sample_name, variable)
            elif plot_type == 'scatter':
                x_var, y_var = variable
                scatter_plot(sample_name, x_var, y_var, KDEplot=not isSave)  # Assumes you want KDE only for view
                if isSave:
                    save_current_fig(f"{sample_name}_{plot_type}_{x_var}_v_{y_var}", plot_type, x_var=x_var, y_var=y_var)
            elif plot_type == 'data_combiner':
                print(sample_name)
                data_combiner(sample_name)
                if isSave:
                    save_current_fig(f"{sample_name}_{variable}_combined", plot_type, variable=variable)
                    print("data saved")
            if viewFig:
                plt.show()

plot_and_save_samples(variables, plot_types, isSave = save_file)