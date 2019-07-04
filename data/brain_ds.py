import fnmatch
import os
import random
import shutil
import string
from collections import defaultdict
from time import sleep

import numpy as np

from batchgenerators.dataloading.data_loader import DataLoaderBase, SlimDataLoaderBase
from batchgenerators.transforms import BrightnessMultiplicativeTransform, BrightnessTransform, GaussianNoiseTransform, \
    MirrorTransform, SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.color_transforms import ClipValueRange
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, PadTransform, FillupPadTransform
from batchgenerators.transforms.noise_transforms import BlankSquareNoiseTransform, GaussianBlurTransform, \
    SquareMaskTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, ZoomTransform
from batchgenerators.transforms.utility_transforms import AddToDictTransform, CopyTransform, NumpyToTensor, \
    ReshapeTransform

from data.data_loader import MultiThreadedDataLoader


def load_dataset(base_dir, pattern='*.npy', slice_offset=0, only_labeled_slices=None, label_slice=None,
                 labeled_threshold=10):
    fls = []
    files_len = []
    slices = []

    for root, dirs, files in os.walk(base_dir):
        for i, filename in enumerate(sorted(fnmatch.filter(files, pattern))):
            npy_file = os.path.join(root, filename)
            numpy_array = np.load(npy_file, mmap_mode="r")

            fls.append(npy_file)
            files_len.append(numpy_array.shape[1])

            if only_labeled_slices is None:

                slices.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])
            else:
                assert label_slice is not None

                for s_idx in range(slice_offset, numpy_array.shape[1] - slice_offset):

                    pixel_sum = np.sum(numpy_array[label_slice, s_idx] > 0.1)
                    if pixel_sum > labeled_threshold:
                        if only_labeled_slices is True:
                            slices.append((i, s_idx))
                    elif pixel_sum == 0:
                        if only_labeled_slices is False:
                            slices.append((i, s_idx))

    return fls, files_len, slices


def get_transforms(mode="train", n_channels=1, target_size=128, add_resize=False, add_noise=False, mask_type="",
                   batch_size=16, rotate=True, elastic_deform=True, rnd_crop=False, color_augment=True):
    tranform_list = []
    noise_list = []

    if mode == "train":

        tranform_list = [FillupPadTransform(min_size=(n_channels, target_size + 5, target_size + 5)),
                         ResizeTransform(target_size=(target_size + 1, target_size + 1),
                                         order=1, concatenate_list=True),

                         # RandomCropTransform(crop_size=(target_size + 5, target_size + 5)),
                         MirrorTransform(axes=(2,)),
                         ReshapeTransform(new_shape=(1, -1, "h", "w")),
                         SpatialTransform(patch_size=(target_size, target_size), random_crop=rnd_crop,
                                          patch_center_dist_from_border=target_size // 2,
                                          do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                          do_rotation=rotate,
                                          angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                          scale=(0.9, 1.2),
                                          border_mode_data="nearest", border_mode_seg="nearest"),
                         ReshapeTransform(new_shape=(batch_size, -1, "h", "w"))]
        if color_augment:
            tranform_list += [  # BrightnessTransform(mu=0, sigma=0.2),
                BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1))]

        tranform_list += [
            GaussianNoiseTransform(noise_variance=(0., 0.05)),
            ClipValueRange(min=-1.5, max=1.5),
        ]

        noise_list = []
        if mask_type == "gaussian":
            noise_list += [GaussianNoiseTransform(noise_variance=(0., 0.2))]


    elif mode == "val":
        tranform_list = [FillupPadTransform(min_size=(n_channels, target_size + 5, target_size + 5)),
                         ResizeTransform(target_size=(target_size + 1, target_size + 1),
                                         order=1, concatenate_list=True),
                         CenterCropTransform(crop_size=(target_size, target_size)),
                         ClipValueRange(min=-1.5, max=1.5),
                         # BrightnessTransform(mu=0, sigma=0.2),
                         # BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1)),
                         CopyTransform({"data": "data_clean"}, copy=True)
                         ]


        noise_list += []

    if add_noise:
        tranform_list = tranform_list + noise_list


    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)


class BrainDataSet(object):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=None, seed=None,
                 num_processes=8, num_cached_per_queue=8 * 4, target_size=128, file_pattern='*.npy',
                 rescale_data=False, add_noise=False, label_slice=None, input_slice=(0,), mask_type="",
                 slice_offset=0, do_reshuffle=True, only_labeled_slices=None, labeled_threshold=10,
                 rotate=True, elastic_deform=True, rnd_crop=True, color_augment=True, tmp_dir=None, use_npz=False,
                 add_slices=0):
        data_loader = BrainDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size,
                                      num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input_slice=input_slice, label_slice=label_slice, slice_offset=slice_offset,
                                      only_labeled_slices=only_labeled_slices, labeled_threshold=labeled_threshold,
                                      tmp_dir=tmp_dir, use_npz=use_npz, add_slices=add_slices)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.n_channels = (add_slices * 2 + 1)
        self.transforms = get_transforms(mode=mode, target_size=target_size, add_resize=rescale_data,
                                         add_noise=add_noise, mask_type=mask_type, batch_size=batch_size,
                                         rotate=rotate, elastic_deform=elastic_deform, rnd_crop=rnd_crop,
                                         color_augment=color_augment, n_channels=self.n_channels)
        self.agumenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.agumenter.restart()
        self.first = True

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.agumenter.renew()
        return iter(self.agumenter)

    # def __next__(self):
    #     return next(self.agumenter)

    def __getitem__(self, index):
        item = self.data_loader[index]
        item = self.transforms(**item)
        return item


class BrainDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=None,
                 seed=None, file_pattern='*.npy', label_slice=None, input_slice=(0,), slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0):

        self.files, self.file_len, self.slices = load_dataset(base_dir=base_dir, pattern=file_pattern,
                                                              slice_offset=slice_offset + add_slices,
                                                              only_labeled_slices=only_labeled_slices,
                                                              label_slice=label_slice,
                                                              labeled_threshold=labeled_threshold)
        super(SlimDataLoaderBase, self).__init__()

        self.batch_size = batch_size
        self.tmp_dir = tmp_dir
        self.use_npz = use_npz
        if self.tmp_dir is not None and self.tmp_dir != "" and self.tmp_dir != "None":
            rnd_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))
            self.tmp_dir = os.path.join(self.tmp_dir, rnd_str)
            if not os.path.exists(self.tmp_dir):
                os.mkdir(self.tmp_dir)

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.slice_idxs = list(range(0, len(self.slices)))
        self.data_len = len(self.slices)
        if num_batches is None:
            self.n_items = self.data_len // self.batch_size
            self.num_batches = self.data_len // self.batch_size
        else:
            self.num_batches = num_batches
            self.n_items = min(self.data_len // self.batch_size, self.num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.add_slices = add_slices

        # print(self.slice_idxs)

        self.np_data = np.asarray(self.slices)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):

        if item > self.n_items:
            raise StopIteration()

        start_idx = (item * self.batch_size) % self.data_len
        stop_idx = ((item + 1) * self.batch_size) % self.data_len

        if stop_idx > start_idx:
            idxs = self.slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()
            idxs = self.slice_idxs[:stop_idx] + self.slice_idxs[start_idx:]

        open_arr = self.np_data[idxs]
        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):

        data = []
        fnames = []
        slice_idxs = []
        labels = []

        for slice in open_array:
            fn_name = self.files[slice[0]]
            slice_idx = slice[1]

            numpy_array = np.load(fn_name, mmap_mode="r")
            numpy_slice = numpy_array[self.input_slice, slice_idx - self.add_slices:slice_idx + self.add_slices + 1, ]

            data.append(numpy_slice)

            if self.label_slice is not None:
                label_slice = numpy_array[self.label_slice,
                              slice_idx - self.add_slices:slice_idx + self.add_slices + 1, ]
                labels.append(label_slice)

            fnames.append(fn_name)
            slice_idxs.append(slice_idx / 200.)
            del numpy_array

        ret_dict = {'data': data, 'fnames': fnames, 'slice_idxs': slice_idxs}
        if self.label_slice is not None:
            ret_dict['seg'] = labels

        return ret_dict


