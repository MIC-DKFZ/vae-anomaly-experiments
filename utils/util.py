import numpy as np
import torch


def f1_score(y_pred, y_label, dims=0, eps=1e-6):
    """Calculates the f1 score of a sample (4d, one-hot encoded) """

    true_positives = torch.sum(y_pred * y_label, dim=dims)
    pos_data = torch.sum(y_label, dim=dims)
    pos_pred = torch.sum(y_pred, dim=dims)
    # false_negatives = torch.sum(torch.sum(y_label, dim=2), dim=2) - true_positives
    # false_positives = torch.sum(torch.sum(y_pred, dim=2), dim=2) - true_positives

    precision = true_positives / (pos_pred + eps)
    recall = true_positives / (pos_data + eps)

    f1_s = 2 * (precision * recall) / (precision + recall + eps)

    # f1_s = (2*true_positives) / (2* true_positves + false_positives + fase_negatives)

    # dices_scores has shape (batch_size, num_classes)
    return f1_s


def calc_hard_dice(x, y, thresh):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = np.asarray(x)

    if isinstance(y, (list, tuple)):
        y = torch.from_numpy(np.asarray(y)).float()

    x_binary = x > thresh
    x_binary = torch.from_numpy(x_binary.astype(int))

    dice = f1_score(x_binary.float(), y.float())

    del x, y, x_binary

    dice = dice.item() if torch.is_tensor(dice) else dice

    return dice


def find_best_val(x, y, val_fn, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75

    val_bottom = val_fn(x, y, q_bottom)
    val_top = val_fn(x, y, q_top)

    if val_bottom > val_top:
        if val_bottom > max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_fn, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top > max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_fn, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)


def smooth_tensor(tensor, kernel_size=8, sigma=3, channels=1):
    # Set these to whatever you want for your gaussian filter

    if kernel_size % 2 == 0:
        kernel_size -= 1

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    import math
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2. * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False,
                                      padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    gaussian_filter.to(tensor.device)

    return gaussian_filter(tensor)


def normalize(tensor):
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta
