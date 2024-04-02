import argparse
import torch
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image


def get_args():
    # --- parsing and configuration --- #
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of VAE for MNIST")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--z-dim', type=int, default=2,
                        help='dimension of hidden variable Z (default: 2)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval between logs about training status (default: 100)')
    parser.add_argument('--learning-rate', type=int, default=1e-3,
                        help='learning rate for Adam optimizer (default: 1e-3)')
    parser.add_argument('--prr', type=bool, default=True,
                        help='Boolean for plot-reproduce-result (default: True')
    parser.add_argument('--prr-z1-range', type=int, default=2,
                        help='z1 range for plot-reproduce-result (default: 2)')
    parser.add_argument('--prr-z2-range', type=int, default=2,
                        help='z2 range for plot-reproduce-result (default: 2)')
    parser.add_argument('--prr-z1-interval', type=int, default=0.2,
                        help='interval of z1 for plot-reproduce-result (default: 0.2)')
    parser.add_argument('--prr-z2-interval', type=int, default=0.2,
                        help='interval of z2 for plot-reproduce-result (default: 0.2)')

    return parser.parse_args()

def get_data_loaders(data_dir, batch_size):
    # --- data loading --- #
    train_data = datasets.FashionMNIST(data_dir, train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(data_dir, train=False,
                            transform=transforms.ToTensor())

    # pin memory provides improved transfer speed
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader


def save_generated_img(image, name, epoch, result_dir, nrow=8):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if epoch % 5 == 0:
        save_path =  os.path.join(result_dir, name+'_'+str(epoch)+'.png')
        save_image(image, save_path, nrow=nrow)