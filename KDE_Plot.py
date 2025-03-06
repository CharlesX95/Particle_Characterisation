# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:38:52 2024

@author: Charlie
"""

import seaborn as sns

plt.figure(figsize=(10, 6))

# Plot KDE for both datasets in the overlap region
sns.kdeplot(data_10x_cropped['Eq Diam'], label='10X', color='blue', fill=True, alpha=0.3)
sns.kdeplot(data_40x_cropped['Eq Diam'], label='40X', color='red', fill=True, alpha=0.3)

plt.xscale('log')
plt.xlabel('Equivalent Diameter (um)')
plt.ylabel('Density')
plt.title('KDE of Particle Size Distributions (10X vs 40X)')
plt.legend()
plt.show()