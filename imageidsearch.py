import numpy as np
import pandas as pd

# Load the .npy file
npy_data = np.load('/home/20074688d/jtt/feature_with_ids.npy')

# Load the .csv file
csv_data = pd.read_csv('XHRpython/merged_output.csv')

# Extract image ids from both files
npy_img_ids = npy_data[:, 0]  # Assuming the first column contains image ids
csv_img_ids = csv_data['img_id'].values

# Filter npy data to keep only rows where the image id appears in the csv file
filtered_npy_data = npy_data[np.isin(npy_img_ids, csv_img_ids)]

# Save the filtered data to a new .npy file
np.save('/home/20074688d/jtt/XHRpython/checked.npy', filtered_npy_data)
