import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
from torch.optim import AdamW
from torch import nn
import matplotlib.pyplot as plt

def load_your_labels():
    # Path to the CSV file
    labels_path = '/home/20074688d/jtt/XHRpython/merged_output.csv'
    # Load the CSV file
    labels_df = pd.read_csv(labels_path)
    # Convert 'y' and 'spurious' columns to numpy array
    labels_y = labels_df['y'].values
    labels_spurious = labels_df['spurious'].map({True: 1, False: 0}).values
    return labels_y, labels_spurious

def augment_data(X, y, y_spurious):
    # Identify minority and majority classes
    minority_class = 1  # 'spurious' is True
    majority_class = 0  # 'spurious' is False

    # Separate majority and minority class samples
    X_minority = X[y_spurious == minority_class]
    X_majority = X[y_spurious == majority_class]
    y_minority = y[y_spurious == minority_class]
    y_majority = y[y_spurious == majority_class]

    # Calculate the number of samples to generate
    num_samples = abs(len(y_majority) - len(y_minority))

    # Generate new samples by adding small random noise to minority class samples
    noise = np.random.normal(0, 0.05, (num_samples, X_minority.shape[1]))
    samples = np.random.choice(len(X_minority), num_samples)
    X_new = X_minority[samples] + noise
    y_new = np.full(num_samples, minority_class)

    # Combine old and new samples
    X_augmented = np.concatenate([X_majority, X_minority, X_new])
    y_augmented = np.concatenate([y_majority, y_minority, y_new])

    return X_augmented, y_augmented, majority_class, minority_class


def main(args):
    # Load your feature space
    X = np.load(args.feature_space)

    # Load your labels
    y, y_spurious = load_your_labels()
    X, y, majority_class, minority_class = augment_data(X, y, y_spurious)
    # Split your data into training and testing sets
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Define your fully connected network
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    )

    # Define your optimizer
    # Define your optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # Define your regularization strength
    l2_lambda = 0.01

    # Train your model
    for epoch in range(args.n_epochs):
        for i in range(0, len(X_train), args.batch_size):
            batch_X = torch.from_numpy(X_train[i:i+args.batch_size])
            batch_y = torch.from_numpy(y_train[i:i+args.batch_size])
            outputs = model(batch_X.float())
            loss = nn.CrossEntropyLoss()(outputs, batch_y.long())

            # Add L2 regularization
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Get softmax probabilities
    # Get softmax probabilities
    softmax = nn.Softmax(dim=1)
    outputs = model(torch.from_numpy(X_test).float())
    probs = softmax(outputs.detach()).numpy()

    # Separate probabilities for each class
    probs_majority = probs[y_test == majority_class, 1]
    probs_minority = probs[y_test == minority_class, 1]

    # Plot probability distributions
    plt.figure(figsize=(10, 6))
    plt.hist(probs_majority, bins=50, alpha=0.5, label='Majority Class')
    plt.hist(probs_minority, bins=50, alpha=0.5, label='Minority Class')
    plt.title('Probability Distributions')
    plt.xlabel('Probability of Being Classified as 1')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig('2.png')

    plt.show()


    # Evaluate your model
    outputs = model(torch.from_numpy(X_test).float())
    predictions = torch.argmax(outputs.detach(), axis=1).numpy()
    accuracy = np.mean(predictions == y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    # Output confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print('Confusion Matrix:')
    print(cm)

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser()

    # Add the necessary arguments
    parser.add_argument("--feature_space", type=str, default="/home/20074688d/jtt/XHRpython/augmented_feature_duiqi.npy")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)