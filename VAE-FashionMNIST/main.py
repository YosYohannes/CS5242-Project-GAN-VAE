'''
Reference: https://github.com/ANLGBOY/VAE-with-PyTorch
'''

import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from model import VAE
from utils import get_args, get_data_loaders, save_generated_img
from train import train_model
from test_model import test_model

args = get_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LOG_INTERVAL = args.log_interval
Z_DIM = args.z_dim
LEARNING_RATE = args.learning_rate
PRR = args.prr
Z1_RANGE = args.prr_z1_range
Z2_RANGE = args.prr_z2_range
Z1_INTERVAL = args.prr_z1_interval
Z2_INTERVAL = args.prr_z2_interval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
result_dir =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'results'))

train_loader, test_loader = get_data_loaders(data_dir, BATCH_SIZE)

model = VAE(Z_DIM).to(device)

def sample_from_model(epoch):
    with torch.no_grad():
        # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
        sample = torch.randn(64, Z_DIM).to(device)
        sample = model.decode(sample).cpu().view(64, 1, 28, 28)
        save_generated_img(sample, 'sample', epoch, result_dir)


def plot_along_axis(epoch):
    z1 = torch.arange(-Z1_RANGE, Z1_RANGE, Z1_INTERVAL).to(device)
    z2 = torch.arange(-Z2_RANGE, Z2_RANGE, Z2_INTERVAL).to(device)
    num_z1 = z1.shape[0]
    num_z2 = z2.shape[0]
    num_z = num_z1 * num_z2

    sample = torch.zeros(num_z, 2).to(device)

    for i in range(num_z1):
        for j in range(num_z2):
            idx = i * num_z2 + j
            sample[idx][0] = z1[i]
            sample[idx][1] = z2[j]

    sample = model.decode(sample).cpu().view(num_z, 1, 28, 28)
    save_generated_img(sample, 'plot_along_z1_and_z2_axis', epoch, result_dir, num_z1)


# --- main function --- #
if __name__ == '__main__':
    train_loss = []
    test_loss = []
    for epoch in range(1, EPOCHS + 1):
        loss_tr = train_model(model, train_loader, epoch, LEARNING_RATE, LOG_INTERVAL, device)
        loss_test = test_model(model, test_loader, epoch, BATCH_SIZE, device, result_dir)
        train_loss.append(loss_tr)
        test_loss.append(loss_test)
        sample_from_model(epoch)

        if PRR:
            plot_along_axis(epoch)

    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.show()