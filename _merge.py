from PIL import Image
import random
from pathlib import Path
import argparse
import numpy as np

from utils import load_image, show_image, save_image, rotate, hflip, vflip, merge


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--save_dir", type=int, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def get_l2_dist(edge1, edge2):
    return np.mean((edge1 - edge2) ** 2)


def edge_idx_to_edge(img, edge_idx):
    if edge_idx == 0:
        return img[: 1, :, :]
    elif edge_idx == 1:
        return img[:, -1:, :]
    elif edge_idx == 2:
        return img[-1:, :, :]
    else:
        return img[:, :1, :]


def load_sub_imgs(sub_imgs_dir):
    sub_imgs = list()
    for img_path in Path(sub_imgs_dir).glob("*.png"):
        sub_img = load_image(img_path)
        sub_h, sub_w, _ = sub_img.shape
        if sub_w < sub_h:
            sub_img = rotate(sub_img)
        sub_imgs.append(sub_img)
    return sub_imgs


def get_sorted_dists(sub_imgs):
    dists = dict()
    for idx1, sub_img1 in enumerate(sub_imgs):
        for idx2, sub_img2 in enumerate(sub_imgs):
            if idx1 >= idx2:
                continue

            for edge_idx1 in range(4):
                for edge_idx2 in range(4):
                    for flip_idx in range(2):
                        edge1 = edge_idx_to_edge(sub_img1, edge_idx1)
                        edge2 = edge_idx_to_edge(sub_img2, edge_idx2)
                        if edge1.shape != edge2.shape:
                            continue

                        if flip_idx == 0:
                            dist = get_l2_dist(edge1, edge2)
                        else:
                            edge_h, edge_w, _ = edge2.shape
                            if edge_h >= edge_w:
                                dist = get_l2_dist(edge1, vflip(edge2))
                            else:
                                dist = get_l2_dist(edge1, hflip(edge2))
                        
                        dists[(idx1, idx2, edge_idx1, edge_idx2, flip_idx)] = dist
    
    sorted_dists = dict(sorted(dists.items(), key=lambda item: item[1]))
    return sorted_dists


def get_coord_of_idx(arr, idx):
    return [i[0] for i in np.where(arr == idx)]


def main():
    sub_imgs = load_sub_imgs("/Users/jongbeomkim/Documents/dmeta")
    sub_w, sub_h, _ = sub_imgs[0].shape

    n_row_splits = 3
    n_col_splits = 2

    sorted_dists = get_sorted_dists(sub_imgs)
    # sorted_dists
    # show_image(sub_imgs[0]), show_image(sub_imgs[4])

    k = 3
    merged = np.empty(
        shape=(sub_h * n_row_splits * k, sub_w * n_col_splits * k, 3), dtype="uint8",
    )
    arr = np.empty(shape=(n_row_splits * k, n_col_splits * k), dtype="uint8")
    center_idx = 5, 3    
    
    new_sub_imgs = list()
    skip_idx_pairs = list()
    for idx1, idx2, edge_idx1, edge_idx2, flip_idx in list(sorted_dists.keys())[: 4]:
        # idx1, idx2, edge_idx1, edge_idx2, flip_idx
        sub_img1 = sub_imgs[idx1]
        sub_img2 = sub_imgs[idx2]
        if flip_idx == 1:
            # sub_img2 = sub_img2[:, :: -1, :]
            sub_img2 = hflip(sub_img2)
            # show_image(sub_img1), show_image(sub_img2)

        edge_idx1, edge_idx2
        if edge_idx1 == 0:
            if edge_idx2 == 0:
                sub_img2 = vflip(sub_img2)
            sub_merged = np.concatenate([sub_img2, sub_img1], axis=0)
        else:
            if edge_idx2 == 1:
                sub_img2 = vflip(sub_img2)
            sub_merged = np.concatenate([sub_img1, sub_img2], axis=0)
        show_image(sub_merged)
        new_sub_imgs.append(sub_merged)

        skip_idx_pairs.append((idx1, idx2))


arr = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])