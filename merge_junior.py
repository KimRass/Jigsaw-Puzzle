from PIL import Image
import random
from pathlib import Path
import argparse
import numpy as np

from utils import load_image, show_image, save_image, rotate, hflip, vflip


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


def load_patchs(patchs_dir):
    patchs = list()
    for img_path in Path(patchs_dir).glob("*.png"):
        patch = load_image(img_path)
        sub_h, sub_w, _ = patch.shape
        if sub_w < sub_h:
            patch = rotate(patch)
        patchs.append(patch)
    return patchs


def get_sorted_dists(patchs):
    dists = dict()
    for idx1, patch1 in enumerate(patchs):
        for idx2, patch2 in enumerate(patchs):
            if idx1 >= idx2:
                continue

            for edge_idx1 in range(4):
                for edge_idx2 in range(4):
                    for flip_idx in range(2):
                        edge1 = edge_idx_to_edge(patch1, edge_idx1)
                        edge2 = edge_idx_to_edge(patch2, edge_idx2)
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


def get_all_variants(img):
    imgs = list()
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    img = rotate(img)
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    return imgs


def merge(patchs, order, n_row_splits, n_col_splits):
    sub_h, sub_w, _ = patchs[0].shape
    merged = np.empty(
        shape=(sub_h * n_row_splits, sub_w * n_col_splits, 3), dtype="uint8",
    )
    for row in range(n_row_splits):
        for col in range(n_col_splits):
            merged[
                row * sub_h: (row + 1) * sub_h,
                col * sub_w: (col + 1) * sub_w,
                :,
            ] = patchs[order[row * n_col_splits + col]]
    return merged


def main():
    from itertools import permutations
    patchs = load_patchs("/Users/jongbeomkim/Documents/dmeta")
    # sub_w, sub_h, _ = patchs[0].shape

    n_row_splits = 2
    n_col_splits = 2
    for order in permutations([0, 1, 2, 3], r=4):
        merged = merge(
            patchs=patchs, order=order, n_row_splits=n_row_splits, n_col_splits=n_col_splits,
        )
        show_image(merged)
