import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 28})


def plot_ftir(folder_path, output_folder="BigDuck", lineweight=8, minor_ticks=5):
    """
    Import and plot FTIR data from headerless CSV files.

    Parameters:
    - folder_path: Path to the folder containing the CSV files.
    - output_folder: Path to save the generated plots (default: 'plots').
    - lineweight: Line thickness for the plots (default: 2).
    - minor_ticks: Number of minor ticks between major ticks (default: 5).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all .csv files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the CSV file (headerless)
            data = pd.read_csv(file_path, header=None)
            wavenumber = data[0]
            intensity = data[1]
            
            # Extract title from the filename
            title = os.path.splitext(filename)[0]
            
            # Plot the data
            plt.figure(figsize=(15.5, 6.25))
            plt.plot(wavenumber, intensity, color = 'black',  linewidth=lineweight)
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Intensity (arb. units)')
            plt.ylim(-0.1, 1.1)
            plt.gca().invert_xaxis()  # Reverse the x-axis
            plt.title(f"FTIR "+title)
            plt.tight_layout(pad=0.3)
            # Add minor ticks and set their frequency
            ax = plt.gca()
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=minor_ticks))
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            
            # Customize tick appearance
            plt.tick_params(which='both', direction='in', top=True, right=True)
            plt.tick_params(which='minor', length=4)  # Minor tick length
            plt.tick_params(which='major', length=8)  # Major tick length
            
            
            
            #plt.show()
            # Save the plot
            output_path = os.path.join(output_folder, f"{title}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
    print(f"Plots saved in the folder: {output_folder}")

# Example usage
folder_path = "./ftir_data_BigDuck"  # Replace with your folder path
plot_ftir(folder_path, lineweight=2, minor_ticks=4)  # Adjust lineweight and minor ticks as needed
