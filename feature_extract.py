import sys
sys.path.append('/home/20074688d/jtt')  # Add the path to the jtt directory

from data.cub_dataset import CUBDataset, custom_collate_fn
import torch
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image
import numpy as np
import os
import pandas as pd

from torchvision import transforms

# Define your data transformation
your_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    # Add more transformations if needed, such as normalization
])


# Function to extract features from the last layer
def extract_features(model, dataloader):
    model.eval()
    features = []
    img_ids = []  # Collect image IDs
    with torch.no_grad():
        for inputs, targets, confounders, ids in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            output = model(inputs)
            features.append(output.cpu())
            img_ids.extend(ids)  # Collect IDs
    return features, img_ids



if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet-50 model checkpoint
    checkpoint_path = "/home/20074688d/jtt/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize ResNet-50 model and load checkpoint weights
    # Initialize ResNet-50 model and load checkpoint weights
    model = resnet50(pretrained=False)

    # Modify the model to remove the last fully connected layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model = model.to(device)
    model.eval()


    # Initialize your custom dataset and dataloader
    your_data_path = "/home/20074688d/jtt/jtt/cub/data/waterbird_complete95_forest2water2"
    metadata_path = os.path.join(your_data_path, "metadata.csv")
    metadata = pd.read_csv('/home/20074688d/jtt/jtt/cub/data/waterbird_complete95_forest2water2/metadata.csv', dtype={'y': 'int', 'place': 'int'})
    confounder_names = ["place"]  # Adjust this based on your actual confounder column name(s)
    target_name = "y"  # Adjust this based on your actual target column name

    # Update: Pass metadata, target_name, and confounder_names
    your_dataset = CUBDataset(metadata, target_name, confounder_names, your_data_path, transform=your_transform)
    dataloader = DataLoader(your_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)  # Adjust parameters

    # Extract features
    extracted_features, img_ids = extract_features(model, dataloader)

    # Process features to remove singleton dimensions
    features_array = np.concatenate([f.numpy() for f in extracted_features], axis=0)
    features_array = np.squeeze(features_array)  # Adjust dimensions

    # Ensure img_ids_array is 2D
    img_ids_array = np.array(img_ids).reshape(-1, 1)

    # Combine IDs and features into a single array
    combined_data = np.hstack((img_ids_array, features_array))

    np.save("/home/20074688d/jtt/feature_with_ids.npy", combined_data)

    print("Feature extraction with image IDs completed.")