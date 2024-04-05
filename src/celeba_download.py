from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


train_set = datasets.CelebA(
    root="./data",
    split="train",
    download=True,
    transform=transforms.Compose([
            ToTensor()
    ])
)

valid_set = datasets.CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=transforms.Compose([
            ToTensor()
    ])
)

test_set = datasets.CelebA(
    root="./data",
    split="test",
    download=True,
    transform=transforms.Compose([
            ToTensor()
    ])
)