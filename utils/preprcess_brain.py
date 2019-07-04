import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import scipy.ndimage as snd
from medpy.io import load


def normalize_img(img, mask):
    xp_mean = np.mean(img[mask])
    xp_std = np.std(img[mask]) + 1e-8

    img[mask] = img[mask] - xp_mean
    img[mask] = img[mask] / xp_std

    return img


if __name__ == '__main__':

    ###### BRATS17
    start_dir = ""
    target_dir = ""

    subj_dict = defaultdict(dict)
    mn_list = []

    for root, dirs, files in os.walk(start_dir):
        for f_name in files:
            if f_name.endswith("seg.nii.gz") or f_name.endswith("t1.nii.gz") or f_name.endswith("t2.nii.gz"):
                print(os.path.join(root, f_name))
                pat_nr = root.split("/")[-1]

                file_name = os.path.join(root, f_name)

                imgs, image_header = load(file_name)
                imgs = imgs[None]

                if f_name.endswith("seg.nii.gz"):
                    imgs = imgs.astype(np.int32)
                else:
                    imgs = imgs.astype(np.float32)

                    x = np.where(imgs > 0.00)

                    img_nn = imgs[x]
                    perc_val = np.percentile(img_nn, 0.05)

                    xp = np.where(imgs > perc_val)

                    xp_mean = np.mean(imgs[xp])
                    xp_std = np.std(imgs[xp]) + 1e-8

                    imgs[x] = imgs[x] - xp_mean
                    imgs[x] = imgs[x] / xp_std

                imgs = imgs.transpose((0, 3, 2, 1))
                imgs = imgs[:, ::-1, :, :]

                imgs = imgs[:, :,
                       imgs.shape[2] // 2 - 95:imgs.shape[2] // 2 + 95,
                       imgs.shape[3] // 2 - 77:imgs.shape[3] // 2 + 78]

                imgs = imgs[:, ::-1, :, :]

                imgs_big = np.zeros((1, 155, 190, 165))
                imgs_big[:, :, :, 5:160] = imgs

                print(imgs.shape)

                if f_name.endswith("t1.nii.gz"):
                    subj_dict[pat_nr]["t1"] = imgs_big[0]
                elif f_name.endswith("t2.nii.gz"):
                    subj_dict[pat_nr]["t2"] = imgs_big[0]
                elif f_name.endswith("seg.nii.gz"):
                    subj_dict[pat_nr]["label"] = imgs_big[0]

    cntr = 1
    for pat_nr, vals in subj_dict.items():
        if "t1" in vals and "t2" in vals and "label" in vals:
            pat_nr = "{:05d}".format(cntr)
            final_array = np.stack((vals["t1"], vals["t2"], vals["label"]))
            target_file = os.path.join(target_dir, pat_nr + ".npy")
            np.save(target_file, final_array)
            cntr += 1
            print(pat_nr)

    print("Brats17 done.")
    exit()

    ###### ISLES 2015 Real

    start_dir = ""
    target_dir = ""

    subj_dict = defaultdict(dict)
    mn_list = []

    for root, dirs, files in os.walk(start_dir):
        for f_name in files:
            if ".MR_T1." in f_name or ".MR_T2." in f_name or "XX.O.OT" in f_name:
                print(os.path.join(root, f_name))
                pat_nr = root[len(start_dir):len(start_dir) + 2].replace("/", "_").lower()

                file_name = os.path.join(root, f_name)

                imgs, image_header = load(file_name)
                imgs = imgs[None]

                if "XX.O.OT" in f_name:
                    imgs = imgs.astype(np.int32)
                else:
                    imgs = imgs.astype(np.float32)

                    x = np.where(imgs > 0.00)

                    img_nn = imgs[x]
                    perc_val = np.percentile(img_nn, 0.05)

                    xp = np.where(imgs > 0.)

                    xp_mean = np.mean(imgs[xp])
                    xp_std = np.std(imgs[xp]) + 1e-8

                    imgs[x] = imgs[x] - xp_mean
                    imgs[x] = imgs[x] / xp_std

                imgs = imgs.transpose((0, 3, 2, 1))

                imgs = imgs[:, :, imgs.shape[2] // 2 - 95:imgs.shape[2] // 2 + 95,
                       imgs.shape[3] // 2 - 77:imgs.shape[3] // 2 + 78]

                imgs = imgs[:, :, ::-1]

                if ".MR_T1." in f_name:
                    subj_dict[pat_nr]["t1"] = imgs[0]
                elif ".MR_T2." in f_name:
                    subj_dict[pat_nr]["t2"] = imgs[0]
                elif "XX.O.OT" in f_name:
                    subj_dict[pat_nr]["label"] = imgs[0]

    for pat_nr, vals in subj_dict.items():
        if "t1" in vals and "t2" in vals and "label" in vals:
            final_array = np.stack((vals["t1"], vals["t2"], vals["label"]))
            target_file = os.path.join(target_dir, pat_nr + ".npy")
            np.save(target_file, final_array)
            print(pat_nr)

    print("Isles15_siss done.")
    exit()

    #### HCP

    start_dir = ""
    target_dir = ""

    t1_templ = "T1w_acpc_dc_restore_brain.nii.gz"
    t2_templ = "T2w_acpc_dc_restore_brain.nii.gz"
    label_templ = "wmparc.nii.gz"

    i = 0

    for subj in os.listdir(start_dir):
        sub_dir = os.path.join(start_dir, subj, "T1w/")
        if os.path.isdir(sub_dir):
            t1_file = os.path.join(sub_dir, t1_templ)
            t2_file = os.path.join(sub_dir, t2_templ)
            label_file = os.path.join(sub_dir, label_templ)

            t1_array = load(t1_file)[0]
            t2_array = load(t2_file)[0]
            label_array = load(label_file)[0]

            t1_array = snd.zoom(t1_array, (0.75, 0.75, 0.75), order=1)
            t1_array = t1_array.transpose((2, 1, 0))
            t1_array = t1_array[:, ::-1, :]
            t1_array = t1_array[0:165, 15:225, 15:180]
            t1_mask = np.where(t1_array != 0)
            normalize_img(t1_array, t1_mask)

            t2_array = snd.zoom(t2_array, (0.75, 0.75, 0.75), order=1)
            t2_array = t2_array.transpose((2, 1, 0))
            t2_array = t2_array[:, ::-1, :]
            t2_array = t2_array[0:165, 15:225, 15:180]

            t2_mask = np.where(t2_array != 0)
            normalize_img(t2_array, t2_mask)

            label_array = snd.zoom(label_array, (0.75, 0.75, 0.75), order=0)
            label_array = label_array.transpose((2, 1, 0))
            label_array = label_array[:, ::-1, :]
            label_array = label_array[0:165, 15:225, 15:180]

            final_array = np.stack((t1_array, t2_array, label_array))
            target_file = os.path.join(target_dir, subj + ".npy")
            np.save(target_file, final_array)

            print(i)
            i += 1

    print("HCP Done.")
    exit()
