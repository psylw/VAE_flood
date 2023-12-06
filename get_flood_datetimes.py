

# %%
import pandas as pd
import matplotlib.pyplot as plt
loc1_stage = pd.read_csv('loc1',sep='\t',engine='python',skiprows=28)

loc2_stage = pd.read_csv('loc2',sep='\t',engine='python',skiprows=28)
#%%

loc1_flow = pd.read_csv('loc1cfs',sep='\t',engine='python',skiprows=30)

loc2_flow  = pd.read_csv('loc2cfs',sep='\t',engine='python',skiprows=30)

loc1_flow['14n'] = pd.to_numeric(loc1_flow['14n'], errors='coerce')
loc2_flow['14n'] = pd.to_numeric(loc2_flow['14n'], errors='coerce')
#%%
loc1_action = 7
loc2_action = 5.4

# %%
# find datetimes where stage above action
times1 = loc1_stage.loc[(loc1_stage['14n']>loc1_action-.05)&(loc1_stage['14n']<loc1_action+.05)]['20d'].values

times2 = loc2_stage.loc[(loc2_stage['14n']>loc2_action-.05)&(loc2_stage['14n']<loc2_action+.05)]['20d'].values
# %%
# find equivalent flow for action stage
loc1_flow.loc[loc1_flow['20d'].isin(times1)]['14n'].hist()

loc1_flow_thr = loc1_flow.loc[loc1_flow['20d'].isin(times1)]['14n'].min()


# %%
# look at stage vs discharge for loc2
curve_flow = loc2_flow.loc[loc2_flow['20d'].isin(loc2_stage['20d'])].dropna()
#curve_flow = curve_flow[curve_flow['14n']>200]

curve_stage = loc2_stage.loc[loc2_stage['20d'].isin(curve_flow['20d'])].dropna()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data
x = curve_flow['14n'] 
y = curve_stage['14n']  # Replace this with your actual data

# Fitting a linear curve (y = mx + c)
m, c = np.polyfit(x, y, 1)  # 1 for linear fit, 2 for quadratic, and so on

x_fit = np.linspace(min(x), max(x)+500, 100)
# Generating y values for the fitted line
y_fit = m * x_fit + c

# Plotting the original data and the fitted line
plt.scatter(x, y, label='Original Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Fitted Linear Curve')
plt.legend()
plt.grid(True)
plt.show()



# %%
# find datetimes where flow above action

loc2_flow_thr = (loc2_action-c)/m 
# %%
loc1_flow_thr = loc1_flow['14n'].quantile(.9)
loc2_flow_thr = loc2_flow['14n'].quantile(.9)
print(loc1_flow_thr)
print(loc2_flow_thr)
loc1pos = loc1_flow.loc[loc1_flow['14n']>loc1_flow_thr]['20d'].values
loc2pos = loc2_flow.loc[loc2_flow['14n']>loc2_flow_thr]['20d'].values

# %%
import glob
filenames_sar = glob.glob('sar//'+'*.tif')

sar_dates1 = []
sar_dates2 = []
for i in range(len(filenames_sar)):
    if filenames_sar[i].split('_')[0][-4:] == 'loc1':
        sar_dates1.append(filenames_sar[i].split('_')[1])
    elif filenames_sar[i].split('_')[0][-4:] == 'loc2':
        sar_dates2.append(filenames_sar[i].split('_')[1])
# %%
flood_dates1 = [loc1pos[i].split()[0] for i in range(len(loc1pos))]
flood_dates2 = [loc2pos[i].split()[0] for i in range(len(loc2pos))]

# %%
loc1flood = set(sar_dates1).intersection(set(flood_dates1))
loc2flood = set(sar_dates2).intersection(set(flood_dates2))
# %%
import os
import shutil

# Source folder containing the files
source_folder = 'sar'

# Destination folder to move the files
destination_folder = 'flood'


    # Iterate through the files and move if filename contains an element from the set
for file_name in filenames_sar:
    for element in loc1flood:
        if element in file_name and file_name.split('_')[0][-4:] == 'loc1':
            # Construct source and destination paths
            source_path = os.path.join(file_name)
            destination_path = os.path.join('sar',destination_folder, file_name[4:])
            
            # Move the file to the destination folder
            shutil.move(source_path, destination_path)


# %%
for file_name in filenames_sar:
    for element in loc2flood:
        if element in file_name and file_name.split('_')[0][-4:] == 'loc2':
            # Construct source and destination paths
            source_path = os.path.join(file_name)
            destination_path = os.path.join('sar',destination_folder, file_name[4:])
            
            # Move the file to the destination folder
            shutil.move(source_path, destination_path)