import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Load the feature space with image IDs
features_with_id = np.load('/home/20074688d/jtt/XHRpython/feature_duiqi.npy')
# Separate the features from the image IDs
image_ids = features_with_id[:, 0]  # First column is image IDs
features = features_with_id[:, 1:]  # The rest are features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=50)  # Adjust the number of components based on your dataset
features_pca = pca.fit_transform(features_scaled)
tsne = TSNE(n_components=2, perplexity=30.0, n_iter=3000, random_state=42)  # t-SNE for 2D visualization
features_tsne = tsne.fit_transform(features_pca)  # Apply on PCA-reduced data for speed & effectiveness
kmeans_pca = KMeans(n_clusters=2, random_state=42)
clusters_pca = kmeans_pca.fit_predict(features_pca)  # Perform clustering on PCA-reduced data
kmeans_tsne = KMeans(n_clusters=2, random_state=42)
clusters_tsne = kmeans_tsne.fit_predict(features_tsne)  # Perform clustering on t-SNE-reduced data

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters_tsne, cmap='viridis', marker='o', alpha=0.5)
plt.title('t-SNE Visualization of Features with K-Means Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar()
plt.show()

# Assuming the CSV file has 'img_id' and 'wrong_1_times' indicating classification correctness
classification_outcomes = pd.read_csv('/home/20074688d/jtt/XHRpython/merged_output.csv')
actual_outcomes = classification_outcomes[['img_id', 'wrong_1_times']].copy()
actual_outcomes['classification_correct'] = (actual_outcomes['wrong_1_times'] == 0).astype(int)
# Create a DataFrame from the image IDs and their corresponding cluster labels
cluster_labels_df = pd.DataFrame({'img_id': image_ids, 'cluster_label': clusters_pca})

# Merge this DataFrame with the actual outcomes DataFrame on 'img_id' to align the data
merged_df = pd.merge(actual_outcomes, cluster_labels_df, on='img_id')
# This is a simplistic approach for illustration. You may need a more sophisticated method to align cluster labels with actual outcomes.
# Assuming cluster_label 0 corresponds to 'correct' and 1 to 'incorrect' based on your analysis
# Note: You might need to adjust this mapping based on the actual distribution of your data

# Flip labels if necessary based on your analysis
merged_df['predicted_correct'] = merged_df['cluster_label'].apply(lambda x: 0 if x == 1 else 1)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(merged_df['classification_correct'], merged_df['predicted_correct'])

print(f'Clustering Accuracy after PCA: {accuracy}')
