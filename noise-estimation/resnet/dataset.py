import torch

# For Dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
import json
import os, sys

file_dir = os.path.dirname(os.path.abspath(__file__))

class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load mapping from synset to integer class
        with open(os.path.join(file_dir, "imagenet_class_index.json")) as f:
            idx_to_label = json.load(f)

        # Create reverse mapping: synset -> class index
        self.synset_to_idx = {v[0]: int(k) for k, v in idx_to_label.items()}

        # Iterate over each class folder
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = self.synset_to_idx.get(class_name, None)
            if label is None:
                continue

            # Iterate over each image file in the class folder
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_path, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def accuracy(outputs, targets, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k"""
    # Get top-1 prediction
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == targets)
    accuracy = correct.float() / targets.size(0)

    # Top-5 accuracy
    _, top5_preds = outputs.topk(5, 1)
    correct_top5 = top5_preds.eq(targets.view(-1,1).expand_as(top5_preds))
    top5_acc = correct_top5.sum().item() / targets.size(0)

    return accuracy, top5_acc

# Imagenet
transform = transforms.Compose([
    transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),  # Resize the shorter side to 256 pixels
    transforms.CenterCrop(224),                                        # Crop the center 224x224 pixels
    transforms.ToTensor(),                                             # Convert the image to a PyTorch tensor
    transforms.Normalize(                                              # Normalize using ImageNet's mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])