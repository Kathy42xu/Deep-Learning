import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from models import model_attributes
from data.data import dataset_attributes
from data.dro_dataset import DRODataset
from utils import get_loader

# Define your process_item function if you have one
def your_process_item_function(item):
    # Your processing logic here
    return item

# Modified get_model function
def get_model(model, pretrained, resume, n_classes, dataset, log_dir):
    if resume:
        model = torch.load(os.path.join(log_dir, "last_model.pth"))
        d = train_data.input_size()[0]
    elif model_attributes[model]["feature_type"] in ("precomputed", "raw_flattened"):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            assert dataset == "MultiNLI"
            from pytorch_transformers import BertConfig, BertForSequenceClassification
            config_class = BertConfig
            model_class = BertForSequenceClassification
            config = config_class.from_pretrained("bert-base-uncased", num_labels=3, finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased", from_tf=False, config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(model, num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else:
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model

# Function for feature extraction
def extract_features(model, dataloader):
    features = []
    labels = []

    with torch.no_grad():
        model.eval()
        for inputs, targets in dataloader:
            # Forward pass to get features
            outputs = model(inputs)
            features.append(outputs)
            labels.append(targets)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels

# Your existing code for preparing data, loading datasets, etc.

# Example: Replace YourDatasetClass with your actual dataset class
train_data = YourDatasetClass(...)  # Initialize your train dataset
val_data = YourDatasetClass(...)    # Initialize your validation dataset

# Get the data loaders
train_loader = get_loader(train_data, train=True, reweight_groups=None, batch_size=32, num_workers=4, pin_memory=True)
val_loader = get_loader(val_data, train=False, reweight_groups=None, batch_size=32, num_workers=4, pin_memory=True)

# Get the ResNet-50 model for feature extraction
model = get_model(model="resnet50", pretrained=True, resume=False, n_classes=2, dataset="CUB", log_dir="./logs")

# Extract features using the train loader
train_features, train_labels = extract_features(model, train_loader)

# Extract features using the validation loader
val_features, val_labels = extract_features(model, val_loader)

# Now, train_features and val_features contain the extracted features for training and validation sets.
# You can use these features for further analysis or downstream tasks.
