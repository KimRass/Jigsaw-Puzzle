from PIL import Image
import random
from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy
import math
# from itertools import permutations
from collections import OrderedDict

from utils import load_image, show_image, save_image, rotate, hflip, vflip, load_patches


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
    return np.mean((edge1 - edge2) ** 2).round(3)


def edge_idx_to_edge(img, edge_idx):
    if edge_idx == 0:
        return img[: 1, :, :]
    elif edge_idx == 1:
        return img[:, -1:, :]
    elif edge_idx == 2:
        return img[-1:, :, :]
    else:
        return img[:, :1, :]


def get_sorted_dists(patches):
    dists = dict()
    for patch_idx1, patch1 in patches.items():
        for patch_idx2, patch2 in patches.items():
            if patch_idx1 >= patch_idx2:
                continue

            best_dist = math.inf
            for edge_idx1 in range(4):
                edge1 = edge_idx_to_edge(patch1, edge_idx1)
                for edge_idx2 in range(4):
                    edge2 = edge_idx_to_edge(patch2, edge_idx2)
                    if edge1.shape != edge2.shape:
                        continue

                    for flip_idx in range(2):
                        if flip_idx == 0:
                            dist = get_l2_dist(edge1, edge2)
                        else:
                            edge_h, edge_w, _ = edge2.shape
                            if edge_h >= edge_w:
                                dist = get_l2_dist(edge1, vflip(edge2))
                            else:
                                dist = get_l2_dist(edge1, hflip(edge2))
                        if dist < best_dist:
                            best_dist = dist
                            best_edge_idx1 = edge_idx1
                            best_edge_idx2 = edge_idx2
                            best_flip_idx = flip_idx
                        dists[(patch_idx1, patch_idx2)] = (
                            best_edge_idx1,
                            best_edge_idx2,
                            best_flip_idx,
                            best_dist,
                        )

    sorted_dists = dict(sorted(dists.items(), key=lambda item: item[1][3]))
    return sorted_dists


def get_coord_of_idx(idx_arr, idx):
    x, y = np.where(idx_arr == idx)
    try:
        return [x[0], y[0]]
    except Exception:
        pass


def get_all_variants(img):
    imgs = list()
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    img = rotate(img)
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    return imgs


# def merge(patches, order, n_row_splits, n_col_splits):
def merge(patches, n_row_splits, n_col_splits):
    sub_h, sub_w, _ = patches[0].shape
    merged = np.empty(
        shape=(sub_h * n_row_splits, sub_w * n_col_splits, 3), dtype="uint8",
    )
    for row in range(n_row_splits):
        for col in range(n_col_splits):
            merged[
                row * sub_h: (row + 1) * sub_h,
                col * sub_w: (col + 1) * sub_w,
                :,
            # ] = patches[order[row * n_col_splits + col]]
            ] = patches[row * n_col_splits + col]
    return merged


def merge_two_patches(patches, idx1, idx2, edge_idx1, edge_idx2, flip_idx):
    patch1 = patches[idx1]
    patch2 = patches[idx2]
    axis = 0 if edge_idx1 in [0, 2] else 1
    if edge_idx1 == edge_idx2:
        if edge_idx2 in [0, 2]:
            patch2 = vflip(patch2)
        else:
            patch2 = hflip(patch2)
    if flip_idx == 1:
        patch2 = hflip(patch2) if axis == 0 else vflip(patch2)
    order = [patch2, patch1] if edge_idx1 in [0, 3] else [patch1, patch2]
    merged = np.concatenate(order, axis=axis)
    return merged


def change_coord(coord, edge_idx):
    if edge_idx == 0:
        coord[0] -= 1
    elif edge_idx == 1:
        coord[1] += 1
    elif edge_idx == 2:
        coord[0] += 1
    else:
        coord[1] -= 1


def fill_arr(idx_arr, coord, idx):
    x, y = coord
    if idx_arr[x, y] == 255:
        idx_arr[x, y] = idx


def transform_dic(dic, idx):
    new_dic = deepcopy(dic)
    for temp_idx in range(idx + 1, len(dic)):
        new_dic[temp_idx + 1] = dic[temp_idx]
    new_dic[idx + 2] = dic[idx]
    return new_dic


def modify_patches(patches, temp_patch_idx1, temp_patch_idx2, flip_idx):
    if temp_patch_idx1 in [1, 3]:
        if temp_patch_idx2 == temp_patch_idx1:
            patches[temp_patch_idx2] = hflip(patches[temp_patch_idx2])
        if flip_idx == 1:
            patches[temp_patch_idx2] = vflip(patches[temp_patch_idx2])
    else:
        if temp_patch_idx2 == temp_patch_idx1:
            patches[temp_patch_idx2] = vflip(patches[temp_patch_idx2])
        if flip_idx == 1:
            patches[temp_patch_idx2] = hflip(patches[temp_patch_idx2])


def main():
    patches = load_patches("/Users/jongbeomkim/Documents/dmeta/3by3")
    sub_h, sub_w, _ = patches[0].shape
    n_row_splits = 3
    n_col_splits = 3
    merged = merge(patches, n_row_splits, n_col_splits)
    show_image(merged)


    # tot_merged = np.empty(
    #     shape=(sub_h * n_row_splits * 2, sub_w * n_col_splits * 2, 3), dtype="uint8",
    # )

    sorted_dists = get_sorted_dists(patches)
    dic = {idx: (*k, *v) for idx, (k, v) in enumerate(sorted_dists.items())}
    dic = {k: v for idx, (k, v) in enumerate(dic.items()) if idx <= 5}
    # dic

    idx_arr = np.full((12, 12), fill_value=255, dtype="uint8")
    # dist_arr = np.zeros((12, 12))
    coord = [5, 5]

    patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist = dic[0]
    fill_arr(idx_arr, coord, patch_idx1)
    change_coord(coord, edge_idx1)
    fill_arr(idx_arr, coord, patch_idx2)
    patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist
    idx_arr

    idx = 1
    while True:
        if (idx_arr != 255).sum() >= n_row_splits * n_col_splits:
            break

        patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist = dic[idx]
        coord1 = get_coord_of_idx(idx_arr, patch_idx1)
        coord2 = get_coord_of_idx(idx_arr, patch_idx2)
        if coord1:
            coord = deepcopy(coord1)
            change_coord(coord, edge_idx1)
            fill_arr(idx_arr, coord, patch_idx2)
        elif coord2:
            coord = deepcopy(coord2)
            change_coord(coord, edge_idx2)
            fill_arr(idx_arr, coord, patch_idx1)
        else:
            # print(patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist)
            dic = transform_dic(dic, idx)
            # break
        idx += 1
        patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist
        idx_arr





# 7 3 5
# 6 4 1
# 8 2 0

# 0 2 8
# 1 4 6
# 5 3 7