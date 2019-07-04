import matplotlib

matplotlib.use("Agg", warn=True)

import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as dist

from torchvision import datasets, transforms
from torchvision.utils import save_image

from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
from trixi.logger import PytorchVisdomLogger
from trixi.util import Config
from trixi.util.pytorchutils import set_seed


class VAE(nn.Module):
    def __init__(self, z=20, input_size=784):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, z)
        self.fc22 = nn.Linear(400, z)
        self.fc3 = nn.Linear(z, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd


def loss_function(recon_x, x, mu, logstd, rec_log_std=0):
    rec_std = math.exp(rec_log_std)
    rec_var = rec_std ** 2

    x_dist = dist.Normal(recon_x, rec_std)
    log_p_x_z = torch.sum(x_dist.log_prob(x), dim=1)

    z_prior = dist.Normal(0, 1.)
    z_post = dist.Normal(mu, torch.exp(logstd))
    kl_div = torch.sum(dist.kl_divergence(z_post, z_prior), dim=1)

    return torch.mean(kl_div - log_p_x_z), kl_div, -log_p_x_z


def train(epoch, model, optimizer, train_loader, device, scaling, vlog, elog, log_var_std):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data_flat = data.flatten(start_dim=1).repeat(1, scaling)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_flat)
        loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            # vlog.show_value(torch.mean(kl).item(), name="Kl-loss", tag="Losses")
            # vlog.show_value(torch.mean(rec).item(), name="Rec-loss", tag="Losses")
            # vlog.show_value(loss.item(), name="Total-loss", tag="Losses")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader, test_loader_abnorm, device, scaling, vlog, elog, image_size, batch_size, log_var_std):
    model.eval()
    test_loss = []
    kl_loss = []
    rec_loss = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data_flat = data.flatten(start_dim=1).repeat(1, scaling)
            recon_batch, mu, logvar = model(data_flat)
            loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
            test_loss += (kl + rec).tolist()
            kl_loss += kl.tolist()
            rec_loss += rec.tolist()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch[:, :image_size].view(batch_size, 1, 28, 28)[:n]])
                # vlog.show_image_grid(comparison.cpu(),   name='reconstruction')

    # vlog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    # vlog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    # vlog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")
    # elog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    # elog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    # elog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")

    test_loss_ab = []
    kl_loss_ab = []
    rec_loss_ab = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader_abnorm):
            data = data.to(device)
            data_flat = data.flatten(start_dim=1).repeat(1, scaling)
            recon_batch, mu, logvar = model(data_flat)
            loss, kl, rec = loss_function(recon_batch, data_flat, mu, logvar, log_var_std)
            test_loss_ab += (kl + rec).tolist()
            kl_loss_ab += kl.tolist()
            rec_loss_ab += rec.tolist()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch[:, :image_size].view(batch_size, 1, 28, 28)[:n]])
                # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction2')

    print('====> Test set loss: {:.4f}'.format(np.mean(test_loss)))

    # vlog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    # vlog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    # vlog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")
    # elog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    # elog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    # elog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")

    kl_roc, kl_pr = elog.get_classification_metrics(kl_loss + kl_loss_ab,
                                                    [0] * len(kl_loss) + [1] * len(kl_loss_ab),
                                                    )[0]
    rec_roc, rec_pr = elog.get_classification_metrics(rec_loss + rec_loss_ab,
                                                      [0] * len(rec_loss) + [1] * len(rec_loss_ab),
                                                      )[0]
    loss_roc, loss_pr = elog.get_classification_metrics(test_loss + test_loss_ab,
                                                        [0] * len(test_loss) + [1] * len(test_loss_ab),
                                                        )[0]

    # vlog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    # vlog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    # vlog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")
    # elog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    # elog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    # elog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")

    # vlog.show_value(np.mean(kl_pr), name="KL-loss", tag="PR")
    # vlog.show_value(np.mean(rec_pr), name="Rec-loss", tag="PR")
    # vlog.show_value(np.mean(loss_pr), name="Total-loss", tag="PR")

    return kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr


def model_run(scaling, batch_size, odd_class, z, seed=123, log_var_std=0, n_epochs=25):
    set_seed(seed)

    config = Config(
        scaling=scaling, batch_size=batch_size, odd_class=odd_class, z=z, seed=seed, log_var_std=log_var_std,
        n_epochs=n_epochs
    )

    image_size = 784
    input_size = image_size * scaling
    device = torch.device("cuda")

    def get_same_index(ds, label, invert=False):
        label_indices = []
        for i in range(len(ds)):
            if invert:
                if ds[i][1] != label:
                    label_indices.append(i)
            if not invert:
                if ds[i][1] == label:
                    label_indices.append(i)
        return label_indices

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_set = datasets.FashionMNIST('/home/david/data/datasets/fashion_mnist', train=True, download=True,
                                      transform=transforms.ToTensor())
    test_set = datasets.FashionMNIST('/home/david/data/datasets/fashion_mnist', train=False,
                                     transform=transforms.ToTensor())

    train_indices_zero = get_same_index(train_set, odd_class, invert=True)
    train_zero_set = torch.utils.data.sampler.SubsetRandomSampler(train_indices_zero)
    test_indices_zero = get_same_index(test_set, odd_class, invert=True)
    test_zero_set = torch.utils.data.sampler.SubsetRandomSampler(test_indices_zero)
    test_indices_ones = get_same_index(test_set, odd_class)
    test_one_set = torch.utils.data.sampler.SubsetRandomSampler(test_indices_ones)

    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_zero_set,
                                               batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_zero_set,
                                              batch_size=batch_size, shuffle=False, **kwargs)
    test_loader_abnorm = torch.utils.data.DataLoader(test_set, sampler=test_one_set,
                                                     batch_size=batch_size, shuffle=False, **kwargs)

    model = VAE(z=z, input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    vlog = PytorchVisdomLogger(exp_name="vae-fmnist")
    elog = PytorchExperimentLogger(base_dir="/home/david/data/logs/mnist_exp_fin", exp_name="fashion-mnist_vae")

    elog.save_config(config, "config")

    for epoch in range(1, n_epochs + 1):
        train(epoch, model, optimizer, train_loader, device, scaling, vlog, elog, log_var_std)

    kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr = test(model, test_loader, test_loader_abnorm, device,
                                                             scaling, vlog, elog,
                                                             image_size, batch_size, log_var_std)

    with open(os.path.join(elog.result_dir, "results.json"), "w") as file_:
        json.dump({
            "kl_roc": kl_roc, "rec_roc": rec_roc, "loss_roc": loss_roc,
            "kl_pr": kl_pr, "rec_pr": rec_pr, "loss_pr": loss_pr,
        }, file_, indent=4)


if __name__ == '__main__':
    scaling = 1
    batch_size = 128
    odd_class = 0
    z = 20
    seed = 123
    log_var_std = 0.

    model_run(scaling, batch_size, odd_class, z, seed, log_var_std)
