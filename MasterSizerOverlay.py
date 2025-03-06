import pandas as pd
import matplotlib.pyplot as plt

# Load the first dataset for the bar chart
nylon_combined_path = "combined_data/PMMA_combined.csv"
nylon_combined = pd.read_csv(nylon_combined_path, header=None)
bin_edges = nylon_combined[0]
counts = nylon_combined[1]

# Convert counts to number density (number fraction)
total_counts = counts.sum()
number_fraction = (counts / total_counts)*100

# Load the second dataset for the curve
nylon_master_path = "MasterSizer_data/PMMA_MasterSizer.csv"
nylon_master = pd.read_csv(nylon_master_path, header=None)
x_master = nylon_master[0]  # Replace with the appropriate column for size
y_master = nylon_master[1]  # Replace with the appropriate column for counts/intensity

# Plot the data
plt.figure(figsize=(8, 6))

# Bar chart for Nylon 6_combined (number density)
bin_width = bin_edges.iloc[1] - bin_edges.iloc[0]  # Assuming uniform bins
plt.bar(bin_edges, number_fraction, width=bin_width, align='center', alpha=0.7, color='skyblue', edgecolor='blue', label="PMMA Combined (Number Density)")

# Curve for Nylon 6_MasterSizer
plt.plot(x_master, y_master, color='red', linewidth=2, label="PMMA MasterSizer")

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.xlim(0.05, 3000)
plt.ylim(0.01, 100)

# Add labels, title, and legend
plt.xlabel('Size (µm) [Log Scale]', fontsize=12)
plt.ylabel('Number Fraction (%) (Log-Scale)', fontsize=12)
plt.title('Overlay of PMMA Size Distributions  (Note Different X Limits)', fontsize=14)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
