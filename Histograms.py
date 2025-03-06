import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Function to calculate the equivalent diameter from Area
def calculate_equivalent_diameter(area_column):
    return 2 * np.sqrt(area_column / np.pi)


# Function to plot histograms from specific variables in CSV files with customizations
def plot_histograms_from_csv(path_to_folder, variables_to_plot=None, x_limits=None, log_y_vars=None, bin_widths=None, log_x_vars=None):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(path_to_folder) if f.endswith('.csv')]
    
    # Initialize variable to hold the variables that will be plotted
    variables_in_use = variables_to_plot
    log_y_vars = log_y_vars or []  # Ensure log_y_vars is a list
    bin_widths = bin_widths or {}  # Ensure bin_widths is a dictionary
    log_x_vars = log_x_vars or []  # Ensure log_x_vars is a list for x-axis log scale
    
    # Add the Equivalent Diameter to variables to plot, if Area is being used
    if 'Area' in variables_in_use and 'Equivalent Diameter' not in variables_in_use:
        variables_in_use.append('Equivalent Diameter')
    
    # Combine data across all CSVs to calculate bin edges for consistency
    bin_edges = {}
    
    # Loop through all CSV files to get global min/max for each variable
    for file in csv_files:
        df = pd.read_csv(os.path.join(path_to_folder, file))
        
        # Add Equivalent Diameter column, calculated from Area
        if 'Area' in df.columns:
            df['Equivalent Diameter'] = calculate_equivalent_diameter(df['Area'])
        
        for column_name in variables_in_use:
            if column_name in df.columns:
                # Remove NaN values for the column
                column_data = df[column_name].dropna()
                
                col_min = column_data.min()
                col_max = column_data.max()
                
                if column_name not in bin_edges:
                    bin_edges[column_name] = [col_min, col_max]
                else:
                    bin_edges[column_name][0] = min(bin_edges[column_name][0], col_min)
                    bin_edges[column_name][1] = max(bin_edges[column_name][1], col_max)
    
    # Calculate consistent bin edges for each variable (adjustable by bin_widths)
    bins_dict = {}
    for var in variables_in_use:
        if var in bin_widths:
            if var in log_x_vars:
                # If x-axis is logarithmic, use logarithmic bins
                bin_width = bin_widths[var]
                bins_dict[var] = np.geomspace(bin_edges[var][0], bin_edges[var][1], int((np.log10(bin_edges[var][1]) - np.log10(bin_edges[var][0])) / bin_width) + 1)
            else:
                # Linear binning
                bin_width = bin_widths[var]
                bins_dict[var] = np.arange(bin_edges[var][0], bin_edges[var][1] + bin_width, bin_width)
        else:
            n_bins = 20  # Default to 20 bins if no bin width is specified
            if var in log_x_vars:
                # Logarithmic bins if x-axis is log
                bins_dict[var] = np.geomspace(bin_edges[var][0], bin_edges[var][1], n_bins + 1)
            else:
                # Linear bins
                bins_dict[var] = np.linspace(bin_edges[var][0], bin_edges[var][1], n_bins + 1)
    
    # Create the figure and axes based on the number of CSV files (rows) and columns (variables to plot)
    num_rows = len(csv_files)
    num_cols = len(variables_in_use)
    
    # Adjust the figure size to have 2:1 aspect ratio for each subplot
    fig_width_per_plot = 8  # Each plot will be 8 units wide (2x wider than tall)
    fig_height_per_plot = 4  # Each plot will be 4 units tall
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * fig_width_per_plot, num_rows * fig_height_per_plot))
    
    # Define distinct colors for each column
    colors = plt.cm.get_cmap('viridis', num_cols)
    
    for row, file in enumerate(csv_files):
        # Read each CSV file into a DataFrame
        df = pd.read_csv(os.path.join(path_to_folder, file))
        
        # Add Equivalent Diameter column if 'Area' is present
        if 'Area' in df.columns:
            df['Equivalent Diameter'] = calculate_equivalent_diameter(df['Area'])
        
        for col, column_name in enumerate(variables_in_use):
            # Select the axis for plotting
            ax = axes[row, col] if len(csv_files) > 1 else axes[col]  # Handle case of 1 row
            
            # Remove NaN values for the column before plotting
            column_data = df[column_name].dropna()
            
            # Plot the histogram with normalization and custom bin widths
            column_data.hist(ax=ax, bins=bins_dict[column_name], density=True, color=colors(col))  # Normalised histogram
            
            # Set x-axis limits if specified
            if x_limits and column_name in x_limits:
                ax.set_xlim(x_limits[column_name])
            
            # Apply logarithmic scaling to the x-axis if the variable is in the log_x_vars list
            if column_name in log_x_vars:
                ax.set_xscale('log')
            
            # Apply logarithmic scaling to the y-axis if the variable is in the log_y_vars list
            if column_name in log_y_vars:
                ax.set_yscale('log')
                
            # Adjust the tick label font size for x-axis and y-axis
            ax.tick_params(axis='x', labelsize=18)  # Set x-axis tick label font size to 12
            ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label font size to 12

            
            # Set titles and labels for outermost subplots only, with increased font size
            if row == 0:  # Top row, set column titles with larger font size
                ax.set_title(column_name, fontsize=32)
            if col == 0:  # Leftmost column, set file name as ylabel with larger font size
                ax.set_ylabel(file, fontsize=24)
    
    # Adjust layout for better visibility
    plt.tight_layout()
    plt.show()

# Specify the path to the folder containing the CSV files
path_to_folder = './10x_data/'  # Updated path

# Define the variables to plot (by column names)
variables_to_plot = ['Area']  # Replace with actual variable names

# Define x-axis limits for specific variables (optional)
x_limits = {
  # Example: setting the x-axis limit for Area
    'Width': (0.099, 3000),
    'Solidity': (0, 1),   # Example: setting the x-axis limit for Solidity
    'Equivalent Diameter': (0.099, 3000),
    'Perim.': (0.99, 30000)# Adding limit for Equivalent Diameter
}

# Specify which variables should have a logarithmic y-axis
log_y_vars = ['Equivalent Diameter', 'Width','AR', 'Perim.']  # Replace with actual variable names that need log scaling

# Specify which variables should have a logarithmic x-axis
#log_x_vars = []  # Example: Add variables where the x-axis should be logarithmic
log_x_vars = ['Width','AR', 'Equivalent Diameter', 'Perim.']
# Specify custom bin widths for certain variables (optional)
bin_widths = {
    'Area': 10,    # Logarithmic bin width (as factor of 10)
    'Width': 0.2,    # Linear bin width
    'Solidity': 0.05, # Linear bin width
    'Equivalent Diameter': 0.2  # Linear bin width
}

# Call the function to plot the histograms with the custom settings
plot_histograms_from_csv(path_to_folder, variables_to_plot, x_limits, log_y_vars, bin_widths, log_x_vars)
