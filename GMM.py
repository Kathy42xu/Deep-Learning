import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the feature space with image IDs
features_with_id = np.load('/home/20074688d/jtt/XHRpython/feature_duiqi.npy')

# Load the CSV file with actual classification outcomes
classification_outcomes = pd.read_csv('/home/20074688d/jtt/XHRpython/merged_output.csv')

# Extract necessary columns and rename for clarity
actual_outcomes = classification_outcomes[['img_id', 'wrong_1_times']].copy()
actual_outcomes.rename(columns={'wrong_1_times': 'classification_correct'}, inplace=True)

# Separate the features and image IDs
image_ids = features_with_id[:, 0]
features = features_with_id[:, 1:]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply Gaussian Mixture Model for clustering
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(features_scaled)
clusters = gmm.predict(features_scaled)

# Attempt to align cluster labels with actual outcomes
# This simplistic approach assumes the first cluster corresponds to correctness; adjust based on your data
predicted_correctness = clusters

# Map image IDs back to their predicted correctness
predicted_df = pd.DataFrame({'img_id': image_ids, 'predicted_correctness': predicted_correctness})

# Merge the predicted outcomes with the actual outcomes to compare
comparison_df = actual_outcomes.merge(predicted_df, on='img_id')

# Calculate the efficiency (accuracy)
accuracy = accuracy_score(comparison_df['classification_correct'], comparison_df['predicted_correctness'])

# Calculate confusion matrix for more insights
conf_matrix = confusion_matrix(comparison_df['classification_correct'], comparison_df['predicted_correctness'])

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
