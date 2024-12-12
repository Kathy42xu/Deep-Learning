import numpy as np

# Replace 'your_file.npy' with the path to your .npy file
file_path = '/home/20074688d/jtt/XHRpython/augmented_feature_duiqi.npy'
# Load the .npy file
data = np.load(file_path)

# Print the structure of the array
print("Shape of the array:", data.shape)

# Print the first few items of the array
# Adjust the slicing according to the array's dimensions
print("First few items:", data[:5])  # This prints the first 5 items
print("Shape of the array:", data.shape)

# If it's a 2D array, this will print the number of rows and columns
if data.ndim == 2:
    print("Size of the array: {} rows x {} columns".format(data.shape[0], data.shape[1]))

# Print the first few items of the array
# Adjust the slicing according to the array's dimensions
print("First few items:", data[:5])  # Adjust the slicing as needed