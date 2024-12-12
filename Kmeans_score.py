import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_score

# Load cluster assignments
cluster_assignments = np.load('/home/20074688d/jtt-master copy/XHRpython/cluster_assignments.npy')  # Adjust the path if needed

# Load metadata
metadata = pd.read_csv('/home/20074688d/jtt-master copy/jtt/cub/data/waterbird_complete95_forest2water2/metadata.csv')

# Ensure the order of images in metadata matches the order in your clustering
# If not, you'll need to sort or rearrange one of them to match the other

# Extract true labels
true_labels = metadata['y'].values

# Calculate metrics
ari = adjusted_rand_score(true_labels, cluster_assignments)
homogeneity = homogeneity_score(true_labels, cluster_assignments)

print(f"Adjusted Rand Index: {ari}")
print(f"Homogeneity Score: {homogeneity}")
