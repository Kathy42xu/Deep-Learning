from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the feature space with image IDs
features_with_id = np.load('/home/20074688d/jtt/XHRpython/feature_duiqi.npy')

# Load the CSV file with actual classification outcomes
classification_outcomes = pd.read_csv('/home/20074688d/jtt/XHRpython/merged_output.csv')

# Extract the necessary columns from the CSV
actual_outcomes = classification_outcomes[['img_id', 'wrong_1_times']].copy()
actual_outcomes.rename(columns={'wrong_1_times': 'classification_correct'}, inplace=True)

# Separate the features and image IDs
image_ids = features_with_id[:, 0]
features = features_with_id[:, 1:]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42).fit(features_scaled)

# Calculate distances of each point to cluster centers
distances = cdist(features_scaled, kmeans.cluster_centers_, 'euclidean')

# Get the distance to the nearest cluster center
min_distances = np.min(distances, axis=1)

# Rank points by their distance to the nearest cluster center
ranked_by_distance = np.argsort(min_distances)[::-1]  # Descending order

# Determine the cutoff for the top 5%
top_5_percent_cutoff = int(len(features_scaled) * 0.056)

# Initialize all assignments to 0 (correct classification)
custom_clusters = np.zeros(len(features_scaled), dtype=int)

# Assign the top 5% to cluster 1 (incorrect classification)
custom_clusters[ranked_by_distance[:top_5_percent_cutoff]] = 1

# Create a DataFrame for the clustering results
predicted_df = pd.DataFrame({
    'img_id': image_ids,
    'custom_predicted_correctness': custom_clusters
})

# Merge with actual outcomes for comparison
comparison_df = pd.merge(actual_outcomes, predicted_df, on='img_id')

# Calculate the accuracy
accuracy = accuracy_score(comparison_df['classification_correct'], comparison_df['custom_predicted_correctness'])

# Calculate the confusion matrix
conf_matrix = confusion_matrix(comparison_df['classification_correct'], comparison_df['custom_predicted_correctness'])

print(f'Custom Clustering Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
