import torch
from train import loss_function
from utils import save_generated_img


def test_model(model, test_loader, epoch, batch_size, device, result_dir):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            cur_loss = loss_function(recon_data, data, mu, logvar).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(batch_size, 8)
                comparison = torch.cat(
                    [data[:num_samples], recon_data.view(batch_size, 1, 28, 28)[:num_samples]]).cpu()
                save_generated_img(comparison, 'reconstruction', epoch, result_dir, num_samples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss
