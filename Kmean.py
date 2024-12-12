import numpy as np
from sklearn.cluster import KMeans

# Load the feature space from the .npy file
features = np.load('/home/20074688d/jtt-master copy/feature.npy')

# Check the shape of the array to confirm it's correct
print(f'Array shape: {features.shape}')
print(f'Array data type: {features.dtype}')

# Perform K-Means clustering
# We are clustering into 2 clusters as per your requirement
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(features)

# Output the result
print(f'Cluster assignments: {clusters}')

# Optionally, save the cluster assignments to a file
np.save('/home/20074688d/jtt-master copy/XHRpython/cluster_assignments.npy', clusters)
