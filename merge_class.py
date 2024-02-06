import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/dmeta-coding_test")

from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy
import math
from itertools import product

from utils import load_patches, hflip, vflip, show_image, save_image


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


class JigsawPuzzleSolver(object):
    def __init__(self, input_dir):

        self.input_dir = input_dir

        self.patches = load_patches(input_dir)

    @staticmethod
    def edge_idx_to_edge(img, edge_idx):
        if edge_idx == 0:
            return img[: 1, :, :]
        elif edge_idx == 1:
            return img[:, -1:, :]
        elif edge_idx == 2:
            return img[-1:, :, :]
        else:
            return img[:, :1, :]

    @staticmethod
    def get_l2_dist(edge1, edge2):
        return np.mean((edge1 - edge2) ** 2).round(3)

    def get_edge_dist(self, patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx):
        edge1 = self.edge_idx_to_edge(img=self.patches[patch_idx1], edge_idx=edge_idx1)
        edge2 = self.edge_idx_to_edge(img=self.patches[patch_idx2], edge_idx=edge_idx2)
        if edge1.shape != edge2.shape:
            return math.inf

        if flip_idx == 0:
            dist = self.get_l2_dist(edge1, edge2)
        else:
            edge_h, edge_w, _ = edge2.shape
            if edge_h >= edge_w:
                dist = self.get_l2_dist(edge1, vflip(edge2))
            else:
                dist = self.get_l2_dist(edge1, hflip(edge2))
        return dist

    def get_best_params(self, patch_idx1, patch_idx2):
        best_dist = math.inf
        for edge_idx1 in range(4):
            for edge_idx2 in range(4):
                for flip_idx in range(2):
                    dist = self.get_edge_dist(
                        patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx,
                    )
                    patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist
                    if dist < best_dist:
                        best_dist = dist
                        best_edge_idx1 = edge_idx1
                        best_edge_idx2 = edge_idx2
                        best_flip_idx = flip_idx
        return best_dist, best_edge_idx1, best_edge_idx2, best_flip_idx

    def get_coord_of_idx(self, patch_idx):
        x, y = np.where(self.idx_arr == patch_idx)
        try:
            return [x[0], y[0]]
        except Exception:
            pass

    def sort_patch_indices_pairs_by_min_dist(self):
        pair_dist = dict()
        for patch_idx1, patch_idx2 in product(self.patches, self.patches):
            if patch_idx1 >= patch_idx2:
                continue

            best_dist, _, _, _ = self.get_best_params(
                patch_idx1=patch_idx1, patch_idx2=patch_idx2)
            pair_dist[(patch_idx1, patch_idx2)] = best_dist

        pair_dist = sorted(pair_dist.items(), key=lambda x: x[1])
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

    def merge_two_patches(self, idx1, idx2, edge_idx1, edge_idx2, flip_idx):
        patch1 = self.patches[idx1]
        patch2 = self.patches[idx2]
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

    def change_coord(self, edge_idx):
        if edge_idx == 0:
            self.coord[0] -= 1
        elif edge_idx == 1:
            self.coord[1] += 1
        elif edge_idx == 2:
            self.coord[0] += 1
        else:
            self.coord[1] -= 1

    def fill_index_arr(self, patch_idx):
        x, y = self.coord
        if patch_idx not in self.idx_arr and self.idx_arr[x, y] == 255:
            self.idx_arr[x, y] = patch_idx

    def transform_pairs(pairs, idx):
        new_pairs = deepcopy(pairs)
        for temp_idx in range(idx + 1, len(pairs)):
            new_pairs[temp_idx + 1] = pairs[temp_idx]
        new_pairs[idx + 2] = pairs[idx]
        return new_pairs

    def modify_patches1(self, patch_idx2, edge_idx1, edge_idx2, flip_idx):
        if edge_idx2 in [1, 3]:
            if edge_idx2 == edge_idx1:
                self.patches[patch_idx2] = hflip(self.patches[patch_idx2])
            if flip_idx == 1:
                self.patches[patch_idx2] = vflip(self.patches[patch_idx2])
        else:
            if edge_idx2 == edge_idx1:
                self.patches[patch_idx2] = vflip(self.patches[patch_idx2])
            if flip_idx == 1:
                self.patches[patch_idx2] = hflip(self.patches[patch_idx2])

    def init_index_array(self, pairs, M, N):
        self.idx_arr = np.full((M * 3, N * 3), fill_value=255, dtype="uint8")
        self.coord = [int(M * 1.5), int(N * 1.5)]

        patch_idx1, patch_idx2 = pairs[0]
        _, best_edge_idx1, best_edge_idx2, best_flip_idx = self.get_best_params(patch_idx1, patch_idx2)

        self.fill_index_arr(patch_idx1)
        self.change_coord(best_edge_idx1)
        self.fill_index_arr(patch_idx2)

        self.modify_patches1(patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)

    def modify_patches2(self, patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx):
        if patch_idx1 in self.idx_arr and patch_idx2 in self.idx_arr:
            return
        
        if patch_idx1 in self.idx_arr:
            trg_patch_idx = patch_idx2
            trg_edge_idx = edge_idx2
        elif patch_idx2 in self.idx_arr:
            trg_patch_idx = patch_idx1
            trg_edge_idx = edge_idx1
        else:
            return

        if trg_edge_idx in [1, 3]:
            if edge_idx2 == edge_idx1:
                self.patches[trg_patch_idx] = hflip(self.patches[trg_patch_idx])
            if flip_idx == 1:
                self.patches[trg_patch_idx] = vflip(self.patches[trg_patch_idx])
        else:
            if edge_idx2 == edge_idx1:
                self.patches[trg_patch_idx] = vflip(self.patches[trg_patch_idx])
            if flip_idx == 1:
                self.patches[trg_patch_idx] = hflip(self.patches[trg_patch_idx])

    def check_termination_flag(self, M, N):
        return (self.idx_arr != 255).sum() >= M * N

    def get_patch_order(self, pairs, M, N):
        idx = 1
        while True:
            if self.check_termination_flag(M=M, N=N):
                break

            patch_idx1, patch_idx2 = pairs[idx]
            _, best_edge_idx1, best_edge_idx2, best_flip_idx = self.get_best_params(patch_idx1, patch_idx2)

            coord1 = self.get_coord_of_idx(patch_idx1)
            coord2 = self.get_coord_of_idx(patch_idx2)
            if coord1:
                self.coord = coord1
                self.modify_patches2(patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
                self.change_coord(best_edge_idx1)
                self.fill_index_arr(patch_idx2)
            elif coord2:
                self.coord = deepcopy(coord2)
                self.modify_patches2(patch_idx1, patch_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
                self.change_coord(best_edge_idx2)
                self.fill_index_arr(patch_idx1)
            else:
                pairs = self.transform_pairs(pairs, idx)
            idx += 1
        order = list(self.idx_arr[(self.idx_arr != 255)])
        return order

    def merge_patches_by_order(self, order, M, N):
        sub_h, sub_w, _ = self.patches[0].shape
        merged = np.empty(
            shape=(sub_h * M, sub_w * N, 3), dtype="uint8",
        )
        for row in range(M):
            for col in range(N):
                merged[
                    row * sub_h: (row + 1) * sub_h,
                    col * sub_w: (col + 1) * sub_w,
                    :,
                ] = self.patches[order[row * N + col]]
        return merged

    def merge(self, M, N):
        pairs = self.sort_patch_indices_pairs_by_min_dist()
        self.init_index_array(pairs=pairs, M=M, N=N)
        order = self.get_patch_order(pairs=pairs, M=M, N=N)
        merged = self.merge_patches_by_order(order=order, M=M, N=N)
        return merged

    def calc_l2_dist_along_cell_edges(self, img, M, N):
        h, w, _ = img.shape
        sub_h = h // M
        sub_w = w // N
        tot_dist = 0
        for row_idx in range(1, M):
            line1 = img[row_idx * sub_h - 1: row_idx * sub_h, :, :]
            line2 = img[row_idx * sub_h: row_idx * sub_h + 1, :, :]
            tot_dist += self.get_l2_dist(line1, line2)
        for col_idx in range(1, N):
            line1 = img[:, col_idx * sub_w - 1: col_idx * sub_w, :]
            line2 = img[:, col_idx * sub_w: col_idx * sub_w + 1, :]
            tot_dist += self.get_l2_dist(line1, line2)
        return tot_dist

    def solve(self, M, N):
        if M == N:
            merged = self.merge(M=M, N=N)
        else:
            merged1 = self.merge(M=M, N=N)
            grid_dist1 = self.calc_l2_dist_along_cell_edges(merged1, M=M, N=N)
            merged2 = self.merge(M=N, N=M)
            grid_dist2 = self.calc_l2_dist_along_cell_edges(merged2, M=N, N=M)
            merged = merged1 if grid_dist1 <= grid_dist2 else merged2
        return merged


def main():
    args = get_args()
    
    if args.M * args.N != len(list(Path(args.INPUT_DIR).glob("*.png"))):
        print("Wrong number of patches!")
    else:
        model = JigsawPuzzleSolver(input_dir=args.INPUT_DIR)
        merged = model.solve(M=args.M, N=args.N)
        # show_image(merged)
        save_image(merged, save_path=args.SAVE_PATH)


if __name__ == "__main__":
    main()
