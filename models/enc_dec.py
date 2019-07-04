import numpy as np
import torch
import torch.nn as nn


class NoOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, conv_op=nn.Conv2d, conv_params=None,
                 normalization_op=nn.BatchNorm2d, normalization_params=None,
                 activation_op=nn.LeakyReLU, activation_params=None):

        super(ConvModule, self).__init__()

        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}
        self.activation_params = activation_params
        if self.activation_params is None:
            self.activation_params = {}
        self.normalization_params = normalization_params
        if self.normalization_params is None:
            self.normalization_params = {}

        self.conv = None
        if conv_op is not None and not isinstance(conv_op, str):
            self.conv = conv_op(in_channels, out_channels, **self.conv_params)

        self.normalization = None
        if normalization_op is not None and not isinstance(normalization_op, str):
            self.normalization = normalization_op(out_channels, **self.normalization_params)

        self.activation = None
        if activation_op is not None and not isinstance(activation_op, str):
            self.activation = activation_op(**self.activation_params)

    def forward(self, input, conv_add_input=None, normalization_add_input=None, activation_add_input=None):

        x = input

        if self.conv is not None:
            if conv_add_input is None:
                x = self.conv(x)
            else:
                x = self.conv(x, **conv_add_input)

        if self.normalization is not None:
            if normalization_add_input is None:
                x = self.normalization(x)
            else:
                x = self.normalization(x, **normalization_add_input)

        if self.activation is not None:
            if activation_add_input is None:
                x = self.activation(x)
            else:
                x = self.activation(x, **activation_add_input)

        return x


# Basic Generator
class Generator(nn.Module):
    def __init__(self, image_size, z_dim=256, h_size=(256, 128, 64),
                 upsample_op=nn.ConvTranspose2d, normalization_op=nn.InstanceNorm2d, activation_op=nn.LeakyReLU,
                 conv_params=None, activation_params=None, block_op=None, block_params=None, to_1x1=True):

        super(Generator, self).__init__()

        if conv_params is None:
            conv_params = {}

        n_channels = image_size[0]
        img_size = np.array([image_size[1], image_size[2]])

        if not isinstance(h_size, list) and not isinstance(h_size, tuple):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        elif len(h_size) < 2:
            raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        # We need to know how many layers we will use at the beginning
        img_size_new = img_size // (2 ** len(h_size))
        if np.min(img_size_new) < 2 and z_dim is not None:
            raise AttributeError("h_size to long, one image dimension has already perished")

        ### Start block
        start_block = []

        # Z_size random numbers

        if not to_1x1:
            kernel_size_start = [min(4, i) for i in img_size_new]
        else:
            kernel_size_start = img_size_new.tolist()

        if z_dim is not None:
            self.start = ConvModule(z_dim, h_size_bot,
                                    conv_op=upsample_op,
                                    conv_params=dict(kernel_size=kernel_size_start, stride=1, padding=0, bias=False,
                                                     **conv_params),
                                    normalization_op=normalization_op,
                                    normalization_params={},
                                    activation_op=activation_op,
                                    activation_params=activation_params
                                    )

            img_size_new = img_size_new * 2
        else:
            self.start = NoOp()

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in h_size[1:]:

            if block_op is not None and not isinstance(block_op, str):
                self.middle_blocks.append(
                    block_op(h_size_bot, **block_params)
                )

            self.middle_blocks.append(
                ConvModule(h_size_bot, h_size_top,
                           conv_op=upsample_op,
                           conv_params=dict(kernel_size=4, stride=2, padding=1, bias=False, **conv_params),
                           normalization_op=normalization_op,
                           normalization_params={},
                           activation_op=activation_op,
                           activation_params=activation_params
                           )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new * 2

        ### End block
        self.end = ConvModule(h_size_bot, n_channels,
                              conv_op=upsample_op,
                              conv_params=dict(kernel_size=4, stride=2, padding=1, bias=False, **conv_params),
                              normalization_op=None,
                              activation_op=None)

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


# Basic Encoder
class Encoder(nn.Module):
    def __init__(self, image_size, z_dim=256, h_size=(64, 128, 256),
                 conv_op=nn.Conv2d, normalization_op=nn.InstanceNorm2d, activation_op=nn.LeakyReLU,
                 conv_params=None, activation_params=None,
                 block_op=None, block_params=None,
                 to_1x1=True):
        super(Encoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        n_channels = image_size[0]
        img_size_new = np.array([image_size[1], image_size[2]])

        if not isinstance(h_size, list) and not isinstance(h_size, tuple):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        # elif len(h_size) < 2:
        #     raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        ### Start block
        self.start = ConvModule(n_channels, h_size_bot,
                                conv_op=conv_op,
                                conv_params=dict(kernel_size=4, stride=2, padding=1, bias=False, **conv_params),
                                normalization_op=normalization_op,
                                normalization_params={},
                                activation_op=activation_op,
                                activation_params=activation_params
                                )
        img_size_new = img_size_new // 2

        ### Middle block (Done until we reach ? x 4 x 4)
        self.middle_blocks = nn.ModuleList()

        for h_size_top in h_size[1:]:

            if block_op is not None and not isinstance(block_op, str):
                self.middle_blocks.append(
                    block_op(h_size_bot, **block_params)
                )

            self.middle_blocks.append(
                ConvModule(h_size_bot, h_size_top,
                           conv_op=conv_op,
                           conv_params=dict(kernel_size=4, stride=2, padding=1, bias=False, **conv_params),
                           normalization_op=normalization_op,
                           normalization_params={},
                           activation_op=activation_op,
                           activation_params=activation_params
                           )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new // 2

            if np.min(img_size_new) < 2 and z_dim is not None:
                raise ("h_size to long, one image dimension has already perished")

        ### End block
        if not to_1x1:
            kernel_size_end = [min(4, i) for i in img_size_new]
        else:
            kernel_size_end = img_size_new.tolist()

        if z_dim is not None:
            self.end = ConvModule(h_size_bot, z_dim,
                                  conv_op=conv_op,
                                  conv_params=dict(kernel_size=kernel_size_end, stride=1, padding=0, bias=False,
                                                   **conv_params),
                                  normalization_op=None,
                                  activation_op=None,
                                  )

            if to_1x1:
                self.output_size = (z_dim, 1, 1)
            else:
                self.output_size = (z_dim, *[i - (j - 1) for i, j in zip(img_size_new, kernel_size_end)])
        else:
            self.end = NoOp()
            self.output_size = img_size_new

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output
