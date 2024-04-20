import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class Afhq(Dataset):
    def __init__(self, train=None, val=None, root_dir='./data/afhq', transforms=transforms.ToTensor()):
        assert not (train and val), "Specify only train or val but not both"
        assert train or val, "Specify train or val"
        if train:
            self.root_dir = root_dir + '/train'
        else:
            self.root_dir = root_dir + '/val'
        self.classes = sorted(os.listdir(self.root_dir))

        self.transforms = transforms

        self.images, self.labels = [], []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        image = self.transforms(image)
        
        return image, label