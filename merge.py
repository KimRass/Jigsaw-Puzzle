import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/dmeta-coding_test")

from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy
import math
from itertools import product
from tqdm import tqdm

from utils import (
    save_image, rotate, hflip, vflip, load_patches, merge_patches, show_image,
)


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def edge_idx_to_edge(img, edge_idx):
    if edge_idx == 0:
        return img[: 1, :, :]
    elif edge_idx == 1:
        return img[:, -1:, :]
    elif edge_idx == 2:
        return img[-1:, :, :]
    else:
        return img[:, :1, :]


def get_l2_dist(edge1, edge2):
    return np.mean((edge1 - edge2) ** 2).round(3)


def get_edge_dist(patches, patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx):
    edge1 = edge_idx_to_edge(img=patches[patch_idx1], edge_idx=edge_idx1)
    edge2 = edge_idx_to_edge(img=patches[patch_idx2], edge_idx=edge_idx2)
    if edge1.shape != edge2.shape:
        return math.inf

    if flip_idx == 0:
        dist = get_l2_dist(edge1, edge2)
    else:
        edge_h, edge_w, _ = edge2.shape
        if edge_h >= edge_w:
            dist = get_l2_dist(edge1, vflip(edge2))
        else:
            dist = get_l2_dist(edge1, hflip(edge2))
    return dist


def get_best_params(patches, patch_idx1, patch_idx2):
    best_dist = math.inf
    for edge_idx1 in range(4):
        for edge_idx2 in range(4):
            for flip_idx in range(2):
                dist = get_edge_dist(
                    patches, patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx,
                )
                patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist
                if dist < best_dist:
                    best_dist = dist
                    best_edge_idx1 = edge_idx1
                    best_edge_idx2 = edge_idx2
                    best_flip_idx = flip_idx
    return best_dist, best_edge_idx1, best_edge_idx2, best_flip_idx


def sort_patch_indices_pairs_by_min_dist(patches):
    pair_dist = dict()
    for patch_idx1, patch_idx2 in product(patches, patches):
        if patch_idx1 >= patch_idx2:
            continue

        best_dist, _, _, _ = get_best_params(
            patches=patches, patch_idx1=patch_idx1, patch_idx2=patch_idx2)
        pair_dist[(patch_idx1, patch_idx2)] = best_dist

    pair_dist = sorted(pair_dist.items(), key=lambda x: x[1])
    print(pair_dist)
    pairs = {idx: i[0] for idx, i in enumerate(pair_dist)}

    patch_idx1, patch_idx2 = pairs[0]
    new_pairs = [(patch_idx1, patch_idx2)]
    visited = {idx: False for idx in range(len(pairs))}
    visited[patch_idx1] = True
    visited[patch_idx2] = True
    pairs[0] = (-1, -1)

    while True:
        min_idx = len(pairs)
        for k in visited.keys():
            if not visited[k]:
                continue
            for idx in range(1, len(pairs)):
                if k in pairs[idx] and idx < min_idx:
                    min_idx = idx
        if min_idx == len(pairs):
            break

        visited[pairs[min_idx][0]] = True
        visited[pairs[min_idx][1]] = True

        new_pairs.append(deepcopy(pairs[min_idx]))
        pairs[min_idx] = (-1, -1)
    return new_pairs


def get_coord_of_idx(idx_arr, idx):
    x, y = np.where(idx_arr == idx)
    try:
        return [x[0], y[0]]
    except Exception:
        pass


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


def fill_arr(idx_arr, coord, patch_idx):
    x, y = coord
    if patch_idx not in idx_arr and idx_arr[x, y] == 255:
        idx_arr[x, y] = patch_idx


def transform_pairs(pairs, idx):
    new_pairs = deepcopy(pairs)
    for temp_idx in range(idx + 1, len(pairs)):
        new_pairs[temp_idx + 1] = pairs[temp_idx]
    new_pairs[idx + 2] = pairs[idx]
    return new_pairs


def modify_patches(patches, idx_arr, patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx):
    # print(idx_arr)
    # print(patch_idx1, patch_idx2)
    if patch_idx1 in idx_arr and patch_idx2 in idx_arr:
        return
    
    # print(patch_idx1 in idx_arr, patch_idx2 in idx_arr)
    if patch_idx1 in idx_arr:
        trg_patch_idx = patch_idx2
        trg_edge_idx = edge_idx2
    elif patch_idx2 in idx_arr:
        trg_patch_idx = patch_idx1
        trg_edge_idx = edge_idx1
    else:
        return

    print(trg_patch_idx)
    if trg_edge_idx in [1, 3]:
        if edge_idx2 == edge_idx1:
            patches[trg_patch_idx] = hflip(patches[trg_patch_idx])
        if flip_idx == 1:
            patches[trg_patch_idx] = vflip(patches[trg_patch_idx])
    else:
        if edge_idx2 == edge_idx1:
            patches[trg_patch_idx] = vflip(patches[trg_patch_idx])
        if flip_idx == 1:
            patches[trg_patch_idx] = hflip(patches[trg_patch_idx])


def init_index_array(patches, pairs, M, N):
    idx_arr = np.full((M * 3, N * 3), fill_value=255, dtype="uint8")
    coord = [int(M * 1.5), int(N * 1.5)]

    patch_idx1, patch_idx2 = pairs[0]
    _, best_edge_idx1, best_edge_idx2, best_flip_idx = get_best_params(patches, patch_idx1, patch_idx2)
    # print(patch_idx1, patch_idx2, _, best_edge_idx1, best_edge_idx2, best_flip_idx)
    # show_image(patches[patch_idx1])
    # show_image(patches[patch_idx2])

    fill_arr(idx_arr=idx_arr, coord=coord, patch_idx=patch_idx1)
    change_coord(coord, best_edge_idx1)
    fill_arr(idx_arr=idx_arr, coord=coord, patch_idx=patch_idx2)

    modify_patches(patches, idx_arr, patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
    print(patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
    print(idx_arr)
    return idx_arr, coord


def check_termination_flag(idx_arr, M, N):
    # print(idx_arr)
    # print((idx_arr != 255).sum())
    return (idx_arr != 255).sum() >= M * N


def get_order(patches, idx_arr, pairs, M, N):
    idx = 1
    # while True:
    for _ in range(4):
        if check_termination_flag(idx_arr=idx_arr, M=M, N=N):
            break

        patch_idx1, patch_idx2 = pairs[idx]
        _, best_edge_idx1, best_edge_idx2, best_flip_idx = get_best_params(patches, patch_idx1, patch_idx2)

        coord1 = get_coord_of_idx(idx_arr, patch_idx1)
        coord2 = get_coord_of_idx(idx_arr, patch_idx2)
        if coord1:
            modify_patches(patches, idx_arr, patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
            coord = deepcopy(coord1)
            change_coord(coord, best_edge_idx1)
            fill_arr(idx_arr=idx_arr, coord=coord, patch_idx=patch_idx2)
            print(patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
            print(idx_arr)
        elif coord2:
            modify_patches(patches, idx_arr, patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
            coord = deepcopy(coord2)
            change_coord(coord, best_edge_idx2)
            fill_arr(idx_arr=idx_arr, coord=coord, patch_idx=patch_idx1)
            print(patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
            print(idx_arr)
        else:
            pairs = transform_pairs(pairs, idx)
        idx += 1
    order = list(idx_arr[(idx_arr != 255)])
    return order


def merge(input_dir, M, N, save_path, idx):
    input_dir = "/Users/jongbeomkim/Documents/dmeta/cut"
    patches = load_patches(input_dir)
    merged = merge_patches(
        patches=patches, order=range(M * N), M=M, N=N,
    )
    show_image(merged)
    pairs = sort_patch_indices_pairs_by_min_dist(patches)

    idx_arr, _ = init_index_array(patches=patches, pairs=pairs, M=M, N=N)
    order = get_order(patches=patches, idx_arr=idx_arr, pairs=pairs, M=M, N=N)
    merged = merge_patches(
        patches=patches, order=order, M=M, N=N,
    )
    save_path = Path(save_path)
    save_image(
        merged,
        save_path=save_path.parent/f"{save_path.stem}-({idx}){save_path.suffix}",
    )
    show_image(merged)


def main():
    args = get_args()
    if args.M * args.N != len(list(Path(args.INPUT_DIR).glob("*.png"))):
        print("Wrong number of patches!")
    else:
        merge(
            input_dir=args.INPUT_DIR, M=args.M, N=args.N, save_path=args.SAVE_PATH, idx=1,
        )
        if args.M != args.N:
            merge(
                input_dir=args.INPUT_DIR, M=args.N, N=args.M, save_path=args.SAVE_PATH, idx=2,
            )


if __name__ == "__main__":
    main()
# [(7, 11), (10, 11), (6, 7), (1, 6), (2, 6), (2, 5), (0, 2), ...]