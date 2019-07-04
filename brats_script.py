import matplotlib

matplotlib.use("Agg", warn=True)

import json
import math
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.distributions as dist
from torch.optim.lr_scheduler import StepLR

from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
from trixi.logger import PytorchVisdomLogger
from trixi.util import Config
from trixi.util.pytorchutils import set_seed

from models.enc_dec import Encoder, Generator
from data.brain_ds import BrainDataSet
from utils.util import smooth_tensor, normalize, find_best_val, calc_hard_dice


class VAE(torch.nn.Module):
    def __init__(self, input_size, h_size, z_dim, to_1x1=True, conv_op=torch.nn.Conv2d,
                 upsample_op=torch.nn.ConvTranspose2d, normalization_op=None, activation_op=torch.nn.LeakyReLU,
                 conv_params=None, activation_params=None, block_op=None, block_params=None, output_channels=None,
                 additional_input_slices=None,
                 *args, **kwargs):

        super(VAE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)
        if output_channels is not None:
            input_size_dec[0] = output_channels
        if additional_input_slices is not None:
            input_size_enc[0] += additional_input_slices * 2

        self.encoder = Encoder(image_size=input_size_enc, h_size=h_size, z_dim=z_dim * 2,
                               normalization_op=normalization_op, to_1x1=to_1x1, conv_op=conv_op,
                               conv_params=conv_params,
                               activation_op=activation_op, activation_params=activation_params, block_op=block_op,
                               block_params=block_params)
        self.decoder = Generator(image_size=input_size_dec, h_size=h_size[::-1], z_dim=z_dim,
                                 normalization_op=normalization_op, to_1x1=to_1x1, upsample_op=upsample_op,
                                 conv_params=conv_params, activation_op=activation_op,
                                 activation_params=activation_params, block_op=block_op,
                                 block_params=block_params)

        self.hidden_size = self.encoder.output_size

    def forward(self, inpt, sample=None, **kwargs):
        enc = self.encoder(inpt, **kwargs)

        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)

        if sample or self.training:
            z = z_dist.rsample()
        else:
            z = mu

        x_rec = self.decoder(z, **kwargs)

        return x_rec, mu, std

    def encode(self, inpt, **kwargs):
        enc = self.encoder(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        return mu, log_std

    def decode(self, inpt, **kwargs):
        x_rec = self.decoder(inpt, **kwargs)
        return x_rec


def loss_function(recon_x, x, mu, logstd, rec_log_std=0, sum_samplewise=True):
    rec_std = math.exp(rec_log_std)
    rec_var = rec_std ** 2

    x_dist = dist.Normal(recon_x, rec_std)
    log_p_x_z = x_dist.log_prob(x)
    if sum_samplewise:
        log_p_x_z = torch.sum(log_p_x_z, dim=(1, 2, 3))

    z_prior = dist.Normal(0, 1.)
    z_post = dist.Normal(mu, torch.exp(logstd))

    kl_div = dist.kl_divergence(z_post, z_prior)
    if sum_samplewise:
        kl_div = torch.sum(kl_div, dim=(1, 2, 3))

    if sum_samplewise:
        loss = torch.mean(kl_div - log_p_x_z)
    else:
        loss = torch.mean(torch.sum(kl_div, dim=(1, 2, 3)) - torch.sum(log_p_x_z, dim=(1, 2, 3)))

    return loss, kl_div, -log_p_x_z


def get_inpt_grad(model, inpt, err_fn):
    model.zero_grad()
    inpt = inpt.detach()
    inpt.requires_grad = True

    err = err_fn(inpt)
    err.backward()

    grad = inpt.grad.detach()

    model.zero_grad()

    return torch.abs(grad.detach())


def train(epoch, model, optimizer, train_loader, device, vlog, elog, log_var_std):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data["data"][0].float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logstd = model(data)
        loss, kl, rec = loss_function(recon_batch, data, mu, logstd, log_var_std)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            vlog.show_value(torch.mean(kl).item(), name="Kl-loss", tag="Losses")
            vlog.show_value(torch.mean(rec).item(), name="Rec-loss", tag="Losses")
            vlog.show_value(loss.item(), name="Total-loss", tag="Losses")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))


def test_slice(model, test_loader, test_loader_abnorm, device, vlog, elog, image_size, batch_size, log_var_std):
    model.eval()
    test_loss = []
    kl_loss = []
    rec_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data["data"][0].float().to(device)
            recon_batch, mu, logstd = model(data)
            loss, kl, rec = loss_function(recon_batch, data, mu, logstd, log_var_std)
            test_loss += (kl + rec).tolist()
            kl_loss += kl.tolist()
            rec_loss += rec.tolist()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                             recon_batch[:n]])
                # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction')

    vlog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    vlog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    vlog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")
    elog.show_value(np.mean(kl_loss), name="Norm-Kl-loss", tag="Anno")
    elog.show_value(np.mean(rec_loss), name="Norm-Rec-loss", tag="Anno")
    elog.show_value(np.mean(test_loss), name="Norm-Total-loss", tag="Anno")

    test_loss_ab = []
    kl_loss_ab = []
    rec_loss_ab = []
    with torch.no_grad():
        for i, data in enumerate(test_loader_abnorm):
            data = data["data"][0].float().to(device)
            recon_batch, mu, logstd = model(data)
            loss, kl, rec = loss_function(recon_batch, data, mu, logstd, log_var_std)
            test_loss_ab += (kl + rec).tolist()
            kl_loss_ab += kl.tolist()
            rec_loss_ab += rec.tolist()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                             recon_batch[:n]])
                # vlog.show_image_grid(comparison.cpu(),                                     name='reconstruction2')

    elog.print('====> Test set loss: {:.4f}'.format(np.mean(test_loss)))

    vlog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    vlog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    vlog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")
    elog.show_value(np.mean(kl_loss_ab), name="Unorm-Kl-loss", tag="Anno")
    elog.show_value(np.mean(rec_loss_ab), name="Unorm-Rec-loss", tag="Anno")
    elog.show_value(np.mean(test_loss_ab), name="Unorm-Total-loss", tag="Anno")

    kl_roc, kl_pr = elog.get_classification_metrics(kl_loss + kl_loss_ab,
                                                    [0] * len(kl_loss) + [1] * len(kl_loss_ab),
                                                    )[0]
    rec_roc, rec_pr = elog.get_classification_metrics(rec_loss + rec_loss_ab,
                                                      [0] * len(rec_loss) + [1] * len(rec_loss_ab),
                                                      )[0]
    loss_roc, loss_pr = elog.get_classification_metrics(test_loss + test_loss_ab,
                                                        [0] * len(test_loss) + [1] * len(test_loss_ab),
                                                        )[0]

    vlog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    vlog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    vlog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")
    elog.show_value(np.mean(kl_roc), name="KL-loss", tag="ROC")
    elog.show_value(np.mean(rec_roc), name="Rec-loss", tag="ROC")
    elog.show_value(np.mean(loss_roc), name="Total-loss", tag="ROC")

    vlog.show_value(np.mean(kl_pr), name="KL-loss", tag="PR")
    vlog.show_value(np.mean(rec_pr), name="Rec-loss", tag="PR")
    vlog.show_value(np.mean(loss_pr), name="Total-loss", tag="PR")

    return kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr, np.mean(test_loss)


def test_pixel(model, test_loader_pixel, device, vlog, elog, image_size, batch_size, log_var_std):
    model.eval()

    test_loss = []
    kl_loss = []
    rec_loss = []

    pixel_class = []
    pixel_rec_err = []
    pixel_grad_all = []
    pixel_grad_kl = []
    pixel_grad_rec = []
    pixel_combi_err = []

    with torch.no_grad():
        for i, data in enumerate(test_loader_pixel):
            inpt = data["data"][0].float().to(device)
            seg = data["seg"].float()[0, :, 0]
            seg_flat = seg.flatten() > 0.5
            pixel_class += seg_flat.tolist()

            recon_batch, mu, logstd = model(inpt)

            loss, kl, rec = loss_function(recon_batch, inpt, mu, logstd, log_var_std, sum_samplewise=False)
            rec = rec.detach().cpu()
            pixel_rec_err += rec.flatten().tolist()

            def __err_fn_all(x, loss_idx=0):  # loss_idx 0: elbo, 1: kl part, 2: rec part
                outpt = model(x)
                recon_batch, mu, logstd = outpt
                loss = loss_function(recon_batch, x, mu, logstd, log_var_std)
                return torch.mean(loss[loss_idx])

            with torch.enable_grad():
                loss_grad_all = get_inpt_grad(model=model, inpt=inpt, err_fn=lambda x: __err_fn_all(x, 0),
                                              ).detach().cpu()
                loss_grad_kl = get_inpt_grad(model=model, inpt=inpt, err_fn=lambda x: __err_fn_all(x, 1),
                                             ).detach().cpu()
                loss_grad_rec = get_inpt_grad(model=model, inpt=inpt, err_fn=lambda x: __err_fn_all(x, 2),
                                              ).detach().cpu()

            pixel_grad_all += smooth_tensor(loss_grad_all).flatten().tolist()
            pixel_grad_kl += smooth_tensor(loss_grad_kl).flatten().tolist()
            pixel_grad_rec += smooth_tensor(loss_grad_rec).flatten().tolist()

            pixel_combi_err += (smooth_tensor(normalize(loss_grad_kl)) * rec).flatten().tolist()

    kl_normalized = np.asarray(pixel_grad_kl)
    kl_normalized = (kl_normalized - np.min(kl_normalized)) / (np.max(kl_normalized) - np.min(kl_normalized))
    rec_normalized = np.asarray(pixel_rec_err)
    rec_normalized = (rec_normalized - np.min(rec_normalized)) / (np.max(rec_normalized) - np.min(rec_normalized))
    combi_add = kl_normalized + rec_normalized

    rec_err_roc, rec_err_pr = elog.get_classification_metrics(pixel_rec_err, pixel_class)[0]
    grad_all_roc, grad_all_pr = elog.get_classification_metrics(pixel_grad_all, pixel_class)[0]
    grad_kl_roc, grad_kl_pr = elog.get_classification_metrics(pixel_grad_kl, pixel_class)[0]
    grad_rec_roc, grad_rec_pr = elog.get_classification_metrics(pixel_grad_rec, pixel_class)[0]
    pixel_combi_roc, pixel_combi_pr = elog.get_classification_metrics(pixel_combi_err, pixel_class)[0]
    add_combi_roc, add_combi_pr = elog.get_classification_metrics(combi_add, pixel_class)[0]

    rec_err_dice, reconst_thres = find_best_val(pixel_rec_err, pixel_class, calc_hard_dice, max_steps=8,
                                                val_range=(0, np.max(pixel_rec_err)))
    grad_kl_dice, grad_kl_thres = find_best_val(pixel_grad_kl, pixel_class, calc_hard_dice, max_steps=8,
                                                val_range=(0, np.max(pixel_grad_kl)))
    pixel_combi_dice, pixel_combi_thres = find_best_val(pixel_combi_err, pixel_class, calc_hard_dice, max_steps=8,
                                                        val_range=(0, np.max(pixel_combi_err)))
    add_combi_dice, _ = find_best_val(combi_add, pixel_class, calc_hard_dice, max_steps=8,
                                      val_range=(0, np.max(combi_add)))

    with open(os.path.join(elog.result_dir, "pixel.json"), "a+") as file_:
        json.dump({
            "rec_err_roc": rec_err_roc, "rec_err_pr": rec_err_pr,
            "grad_all_roc": grad_all_roc, "grad_all_pr": grad_all_pr,
            "grad_kl_roc": grad_kl_roc, "grad_kl_pr": grad_kl_pr,
            "grad_rec_roc": grad_rec_roc, "grad_rec_pr": grad_rec_pr,
            "pixel_combi_roc": pixel_combi_roc, "pixel_combi_pr": pixel_combi_pr,
            "rec_err_dice": rec_err_dice, "grad_kl_dice": grad_kl_dice, "pixel_combi_dice":
                pixel_combi_dice,

        }, file_, indent=4)


def model_run(patch_size, batch_size, odd_class, z, seed=123, log_var_std=0, n_epochs=5,
              model_h_size=(16, 32, 64, 256), exp_name="exp", folder_name="exp"):
    set_seed(seed)

    config = Config(
        patch_size=patch_size, batch_size=batch_size, odd_class=odd_class, z=z, seed=seed, log_var_std=log_var_std,
        n_epochs=n_epochs
    )

    device = torch.device("cuda")

    datasets_common_args = {
        "batch_size": batch_size,
        "target_size": patch_size,
        "input_slice": [1, ],
        "add_noise": True,
        "mask_type": "gaussian",  # 0.0, ## TODO
        "elastic_deform": False,
        "rnd_crop": True,
        "rotate": True,
        "color_augment": True,
        "add_slices": 0,
    }

    input_shape = (
        datasets_common_args["batch_size"], 1, datasets_common_args["target_size"], datasets_common_args["target_size"])

    train_set_args = {
        "base_dir": "hcp/",
        # "num_batches": 500,
        "slice_offset": 20,
        "num_processes": 8,
    }
    test_set_normal_args = {
        "base_dir": "brats17/",
        # "num_batches": 100,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
        "only_labeled_slices": False,
    }
    test_set_unormal_args = {
        "base_dir": "brats17/",
        # "num_batches": 100,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
        "only_labeled_slices": True,
        "labeled_threshold": 10,
    }
    test_set_all_args = {
        "base_dir": "brats17_test/",
        # "num_batches": 50,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 2,
        "slice_offset": 20,
        "label_slice": 2,
    }

    train_loader = BrainDataSet(**datasets_common_args, **train_set_args)
    test_loader_normal = BrainDataSet(**datasets_common_args, **test_set_normal_args)
    test_loader_abnorm = BrainDataSet(**datasets_common_args, **test_set_unormal_args)
    test_loader_all = BrainDataSet(**datasets_common_args, **test_set_all_args)

    model = VAE(input_size=input_shape[1:], h_size=model_h_size, z_dim=z).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=1)

    vlog = PytorchVisdomLogger(exp_name=exp_name)
    elog = PytorchExperimentLogger(base_dir=folder_name, exp_name=exp_name)

    elog.save_config(config, "config")

    for epoch in range(1, n_epochs + 1):
        train(epoch, model, optimizer, train_loader, device, vlog, elog, log_var_std)

    kl_roc, rec_roc, loss_roc, kl_pr, rec_pr, loss_pr, test_loss = test_slice(model, test_loader_normal,
                                                                              test_loader_abnorm, device,
                                                                              vlog, elog, input_shape, batch_size,
                                                                              log_var_std)

    with open(os.path.join(elog.result_dir, "results.json"), "w") as file_:
        json.dump({
            "kl_roc": kl_roc, "rec_roc": rec_roc, "loss_roc": loss_roc,
            "kl_pr": kl_pr, "rec_pr": rec_pr, "loss_pr": loss_pr,
        }, file_, indent=4)

    elog.save_model(model, "vae")

    test_pixel(model, test_loader_all, device, vlog, elog, input_shape, batch_size, log_var_std)

    print("All done....")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    patch_size = 64
    batch_size = 64
    odd_class = 0
    z = 256
    seed = 123
    log_var_std = 0.

    model_run(patch_size, batch_size, odd_class, z, seed, log_var_std)
