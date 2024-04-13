import os
import argparse
from models.model_settings import BASE_DIR
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

PATH_DIR = BASE_DIR + '/custom_classifier'

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='Classify sample images.')
  parser.add_argument('-a', '--attribute', type=str, required=True,
                      help='Attribute to classify images on. (required)')
  parser.add_argument('-i', '--input_dir', type=str, required=True,
                      help='Directory of sample images to classify(required)')

  return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    transform = Compose([
        Resize((224, 224)),  # Resize to 224x224
        ToTensor(),          # Convert to tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('jpg'):
                image_files.append(os.path.join(root, file))
    image_files = sorted(image_files)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification, so output size is 1
    model.load_state_dict(torch.load(os.path.join(PATH_DIR, f'resnet18_{args.attribute}.pth')))

    model.eval()
    labels = []
    with torch.no_grad():
        with tqdm(image_files, unit="image") as loader:
            for input in loader:
                loader.set_description(f"classifying")
                img = Image.open(input)
                data = transform(img)
                output = model(data.unsqueeze(0))
                labels.append(output.item())
    labels = np.array(labels).reshape(-1, 1)
    labels = (labels > 0.5).astype(int)
    np.save(os.path.join(args.input_dir, f'{args.attribute}_scores.npy'), labels)

if __name__ == '__main__':
  main()