# test_dataloader.py
import os
from data.cub_dataset import CUBDataset, custom_collate_fn
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms


# Replace these paths with the correct paths in your environment
metadata_path = "/home/20074688d/jtt-master copy/jtt/cub/data/waterbird_complete95_forest2water2/metadata.csv"
data_dir = "/home/20074688d/jtt-master copy/jtt/cub/data/waterbird_complete95_forest2water2"

# Read your metadata
metadata = pd.read_csv(metadata_path)
confounder_names = ["place"]  # Adjust to your actual confounder column name(s)
target_name = "y"  # Adjust to your actual target column name

# Define your transform
# Replace with the actual transform you are using
your_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# Initialize your custom dataset
your_dataset = CUBDataset(metadata, target_name, confounder_names, data_dir, transform=your_transform)

# Initialize DataLoader with the custom collate function
test_loader = DataLoader(your_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Iterate over the DataLoader
for data in test_loader:
    print(data)
    break  # Just to check the first batch
