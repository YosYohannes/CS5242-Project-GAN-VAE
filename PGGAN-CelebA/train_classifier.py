import os
import argparse
from models.model_settings import BASE_DIR
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.model_selection import train_test_split

target_attr = 'Eyeglasses'

PATH_DIR = BASE_DIR + '/custom_classifier'

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='Classify sample images.')
  parser.add_argument('-a', '--attribute', type=str, required=True,
                      help='Attribute to classify images on. (required)')
  parser.add_argument('-i', '--input_dir', type=str, required=True,
                      help='Directory of dataset (required)')
  parser.add_argument('-e', '--epoch', type=int, default=10,
                      help='No of epoch to train (optional)')

  return parser.parse_args()


transform = Compose([
    Resize((224, 224)),  # Resize to 224x224
    ToTensor(),          # Convert to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class CustomDataset(Dataset):
    def __init__(self, dataframe, datafolder):
        self.dataframe = dataframe.replace(-1, 0)
        self.datafolder = datafolder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = os.path.join(self.datafolder, self.dataframe.iloc[idx, 0])
        img = Image.open(file_path)
        data = transform(img)
        label = self.dataframe.iloc[idx, 1]
        return data, torch.tensor(label)


def get_balanced_dataset(labelfile, target_attr):
  df = pd.read_csv(labelfile, sep=' ', header=0)
  df = df[['filename', target_attr]]
  min_count = df[target_attr].value_counts().min()
  grouped = df.groupby(target_attr)
  return grouped.apply(lambda x: x.sample(min_count)).reset_index(drop=True)

def main():
    args = parse_args()
    df = get_balanced_dataset('celeba_labels.txt', args.attribute)
    train_dataset, test_dataset = train_test_split(df, test_size=0.2)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)

    train_dataset = CustomDataset(train_dataset, args.input_dir)
    test_dataset = CustomDataset(test_dataset, args.input_dir)
    val_dataset = CustomDataset(val_dataset, args.input_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Define your model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification, so output size is 1
    model = model.cuda()  # Move model to GPU if available

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(args.epoch):
        with tqdm(train_loader, unit="batch") as tepoch:
            model.train()
            train_loss = 0
            total_correct = 0
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                total_correct += torch.sum((outputs.squeeze() > 0.5) == labels)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                train_loss += loss
                optimizer.step()
            total_samples = len(train_loader.dataset)
            print(f'Training loss = {train_loss / total_samples}, acc = {total_correct / total_samples}')

        with tqdm(val_loader, unit="batch") as tepoch:
            model.eval()
            val_loss = 0
            total_correct = 0
            with torch.no_grad():
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    total_correct += torch.sum((outputs.squeeze() > 0.5) == labels)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss
            total_samples = len(val_loader.dataset)
        print(f'Validation loss = {val_loss / total_samples}, acc = {total_correct / total_samples}')
        torch.save(model.state_dict(), os.path.join(PATH_DIR, f'resnet_18_{args.attribute}_e_{epoch}_acc_{(total_correct / total_samples):.4f}.pth'))

    # Final test
    with tqdm(test_loader, unit="batch") as tepoch:
        model.eval()
        test_loss = 0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                total_correct += torch.sum((outputs.squeeze() > 0.5) == labels)
                loss = criterion(outputs.squeeze(), labels.float())
                test_loss += loss
    total_samples = len(test_loader.dataset)
    print(f'Test loss = {test_loss / total_samples}, acc = {total_correct / total_samples}')

if __name__ == '__main__':
  main()