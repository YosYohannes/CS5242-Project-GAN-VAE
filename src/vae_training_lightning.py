import torch
import torch.nn.functional as F
import zipfile
import os
from tqdm import tqdm
from models.CelebVariationalAutoencoder import CelebVariationalAutoencoder
from utils.FaceDataset import FaceDataset

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import lightning as pl

num_epochs = 50
batch_size = 512
lr = 0.0005
train_ratio = 0.85
val_ratio = 0.10
test_ratio = 1 - train_ratio - val_ratio

dataset = FaceDataset("./data/celeba/img_align_celeba/")
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
val_set, test_set = torch.utils.data.random_split(val_set, [val_size, test_size])
train_dl = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=64)
val_dl = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=64)
test_dl = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=64)

# device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model = CelebVariationalAutoencoder(train_set[0][0][None], in_c=3, enc_out_c=[32, 64, 64, 64],
                               enc_ks=[3, 3, 3, 3], enc_pads=[1, 1, 0, 1], enc_strides=[1, 2, 2, 1],
                               dec_out_c=[64, 64, 32, 3], dec_ks=[3, 3, 3, 3], dec_strides=[1, 2, 2, 1],
                               dec_pads=[1, 0, 1, 1], dec_op_pads=[0, 1, 1, 0], z_dim=200)
# model.cuda(device)
# model.train()

def vae_kl_loss(mu, log_var):
    return -.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

def vae_loss(y_pred, mu, log_var, y_true, r_loss_factor=1000):
    r_loss = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
    kl_loss = vae_kl_loss(mu, log_var)
    return r_loss_factor * r_loss + kl_loss

class CelebVAE(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model

        self.lr = lr
        self.loss = vae_loss

        self.save_hyperparameters(ignore=["model"])

    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        pred, mu, log_var = self.forward(data)
        loss = self.loss(pred, mu, log_var, data)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred, mu, log_var = self.forward(data)
        loss = self.loss(pred, mu, log_var, data)
        self.log_dict({"val_loss": loss}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr / (self.current_epoch * 2 + 1), betas=(.9, .99), weight_decay=1e-2)
    
net = CelebVAE(model=model, lr=lr)

tblogger = TensorBoardLogger("tb_logs", name="vae_celeba")
csvlogger = CSVLogger("csv_logs", name="vae_celeba_csv")

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="{epoch:02d}_{val_loss:.1f}",
)

# earlystopping_callback = EarlyStopping(
#     monitor = "val_loss",
#     min_delta = 0.05,  # use a reasonable value for your training
#     patience = 3,  # set check_val_every_n_epoch in Trainer
#     mode = "max",  # we are monitoring accuracy so choose "max"
# )

trainer = pl.Trainer(
    max_epochs = num_epochs,
    accelerator = 'auto',
    devices = 1,
    gradient_clip_val = 0.25, gradient_clip_algorithm = "norm",
    logger = (tblogger, csvlogger),
    log_every_n_steps=50,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=1
)

trainer.fit(
    model = net,
    train_dataloaders = train_dl,
    val_dataloaders = val_dl
)