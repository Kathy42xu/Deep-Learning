import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CSV file into a DataFrame
df = pd.read_csv('XHRpython/merged_output.csv')

# Compare the 'y' and 'space' columns and create the 'minority' column
df['minority'] = (df['y'] != df['place']).astype(int)

# Create a new column 'wrong' that indicates whether a mistake was made
df['wrong'] = (df['wrong_1_times'] == 1).astype(int)

# Calculate the confusion matrix for the minority column and the wrong column
conf_matrix = confusion_matrix(df['minority'], df['wrong'])

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Save the modified DataFrame back to a CSV file
df.to_csv('merged_output_modified.csv', index=False)
# Calculate the total number of mistakes made by the minority and majority groups
total_minority_mistakes = df[(df['minority'] == 1) & (df['wrong'] == 1)].shape[0]
total_majority_mistakes = df[(df['minority'] == 0) & (df['wrong'] == 1)].shape[0]

# Calculate the total number of instances in the minority and majority groups
total_minority = df[df['minority'] == 1].shape[0]
total_majority = df[df['minority'] == 0].shape[0]

# Calculate the proportion of mistakes made by each group
proportion_minority_mistakes = total_minority_mistakes / total_minority if total_minority > 0 else 0
proportion_majority_mistakes = total_majority_mistakes / total_majority if total_majority > 0 else 0

# Print the proportions
print(f"Proportion of mistakes made by the minority group: {proportion_minority_mistakes:.2f}")
print(f"Proportion of mistakes made by the majority group: {proportion_majority_mistakes:.2f}")
