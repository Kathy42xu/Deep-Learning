import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim import AdamW
from torch import nn

def load_your_labels():
    # Path to the CSV file
    labels_path = '/home/20074688d/jtt/XHRpython/merged_output.csv'
    # Load the CSV file
    labels_df = pd.read_csv(labels_path)
    # Extract labels from the 'y' column
    labels = labels_df['y'].values
    return labels

def main(args):
    # Load your feature space
    X = np.load(args.feature_space)

    # Load your labels
    y = load_your_labels()

    # Split your data into training and testing sets
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
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train your model
    for epoch in range(args.n_epochs):
        for i in range(0, len(X_train), args.batch_size):
            batch_X = torch.from_numpy(X_train[i:i+args.batch_size])
            batch_y = torch.from_numpy(y_train[i:i+args.batch_size])
            outputs = model(batch_X.float())
            loss = nn.CrossEntropyLoss()(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Evaluate your model
    outputs = model(torch.from_numpy(X_test).float())
    predictions = torch.argmax(outputs.detach(), axis=1).numpy()
    accuracy = np.mean(predictions == y_test)
    print('Accuracy: %.2f' % (accuracy*100))

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