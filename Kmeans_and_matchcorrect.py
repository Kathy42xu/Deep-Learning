import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Map image IDs back to their predicted correctness
predicted_df = pd.DataFrame({'img_id': image_ids, 'predicted_correctness': clusters})

# Merge the predicted outcomes with the actual outcomes to compare
comparison_df = actual_outcomes.merge(predicted_df, on='img_id')

# Ensure classification_correct is binary and matches predicted_correctness
# Here, you might need to map predicted_correctness or classification_correct correctly
# For simplicity, ensure they are both set for comparison as binary outcomes directly

# Calculate the efficiency (accuracy)
accuracy = accuracy_score(comparison_df['classification_correct'], comparison_df['predicted_correctness'])

# Calculate confusion matrix for more insights
conf_matrix = confusion_matrix(comparison_df['classification_correct'], comparison_df['predicted_correctness'])

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Output the cluster results to a CSV file
predicted_df.to_csv('/home/20074688d/jtt/XHRpython/cluster_results.csv', index=False)
