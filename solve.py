from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy
import math
from itertools import product
import time
from datetime import timedelta

from utils import load_pieces, hflip, vflip, save_image


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
    @staticmethod
    def get_elapsed_time(start_time):
        """
        경과된 시간을 측정합니다.
        """
        return timedelta(seconds=round(time.time() - start_time))

    @staticmethod
    def edge_idx_to_edge(img, edge_idx):
        """
        `edge_idx`는 0, 1, 2, 3 중 하나를 가질 수 있으며 각각 `img`의 상우하좌 Edge를 반환합니다.
        """
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
        """
        두 개의 Edge 사이의 L2 distance를 반환합니다.
        """
        return np.mean((edge1 - edge2) ** 2).round(3)

    def get_edge_dist(self, piece_idx1, piece_idx2, edge_idx1, edge_idx2, flip_idx):
        """
        `piece_idx1`, `piece_idx2`는 `self.pieces`에 대한 인덱스를 나타냅니다.
        `flip_idx`는 두 개의 조각들 중 어느 하나를 둘 간의 접촉면에 대해서 반대로 돌릴지 여부를 나타냅니다.
        두 개의 조각들에 대해서 각 조각의 Edge를 추출한 후 L2 distance를 반환합니다.
        """
        edge1 = self.edge_idx_to_edge(img=self.pieces[piece_idx1], edge_idx=edge_idx1)
        edge2 = self.edge_idx_to_edge(img=self.pieces[piece_idx2], edge_idx=edge_idx2)
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

    def get_best_params(self, piece_idx1, piece_idx2):
        """
        두 개의 조각에서 각각 Edge를 추출하여 L2 distance를 비교하는 8개의 경우의 수 중에서
        가장 그 값이 작게 되는 경우의 Parameters를 반환합니다.
        """
        best_dist = math.inf
        for edge_idx1 in range(4):
            for edge_idx2 in range(4):
                for flip_idx in range(2):
                    dist = self.get_edge_dist(
                        piece_idx1, piece_idx2, edge_idx1, edge_idx2, flip_idx,
                    )
                    piece_idx1, piece_idx2, edge_idx1, edge_idx2, flip_idx, dist
                    if dist < best_dist:
                        best_dist = dist
                        best_edge_idx1 = edge_idx1
                        best_edge_idx2 = edge_idx2
                        best_flip_idx = flip_idx
        return best_dist, best_edge_idx1, best_edge_idx2, best_flip_idx

    def get_coord_of_idx(self, piece_idx):
        """
        `self.idx_arr`에서 `piece_idx`가 위치한 곳의 인덱스를 반환합니다.
        """
        x, y = np.where(self.idx_arr == piece_idx)
        try:
            return [x[0], y[0]]
        except Exception:
            pass

    def sort_piece_indices_pairs_by_min_dist(self):
        """
        조각의 인덱스 쌍을 각각에 대응하는 서로 다른 두 조각 사이에서 발생 가능한 가장 작은
        Edges 간의 L2 distance에 따라서 정렬합니다.
        """
        pair_dist = dict()
        for piece_idx1, piece_idx2 in product(self.pieces, self.pieces):
            if piece_idx1 >= piece_idx2:
                continue

            best_dist, _, _, _ = self.get_best_params(
                piece_idx1=piece_idx1, piece_idx2=piece_idx2)
            pair_dist[(piece_idx1, piece_idx2)] = best_dist

        pair_dist = sorted(pair_dist.items(), key=lambda x: x[1])
        pairs = {idx: i[0] for idx, i in enumerate(pair_dist)}

        piece_idx1, piece_idx2 = pairs[0]
        new_pairs = [(piece_idx1, piece_idx2)]
        visited = {idx: False for idx in range(len(pairs))}
        visited[piece_idx1] = True
        visited[piece_idx2] = True
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

    def fill_index_arr(self, piece_idx):
        """
        `self.idx_arr`에서 `self.coord`의 좌표에 `piece_idx`를 삽입합니다.
        """
        x, y = self.coord
        if piece_idx not in self.idx_arr and self.idx_arr[x, y] == 255:
            self.idx_arr[x, y] = piece_idx

    def change_coord(self, edge_idx):
        """
        `edge_idx`에 따라 `self.coord`로부터 상우하좌 중 한 방향으로 한 칸 이동합니다.
        """
        if edge_idx == 0:
            self.coord[0] -= 1
        elif edge_idx == 1:
            self.coord[1] += 1
        elif edge_idx == 2:
            self.coord[0] += 1
        else:
            self.coord[1] -= 1

    def modify_pieces1(self, piece_idx2, edge_idx1, edge_idx2, flip_idx):
        if edge_idx2 in [1, 3]:
            if edge_idx2 == edge_idx1:
                self.pieces[piece_idx2] = hflip(self.pieces[piece_idx2])
            if flip_idx == 1:
                self.pieces[piece_idx2] = vflip(self.pieces[piece_idx2])
        else:
            if edge_idx2 == edge_idx1:
                self.pieces[piece_idx2] = vflip(self.pieces[piece_idx2])
            if flip_idx == 1:
                self.pieces[piece_idx2] = hflip(self.pieces[piece_idx2])

    def init_index_array(self, pairs, M, N):
        self.idx_arr = np.full((M * 3, N * 3), fill_value=255, dtype="uint8")
        self.coord = [int(M * 1.5), int(N * 1.5)]

        piece_idx1, piece_idx2 = pairs[0]
        _, best_edge_idx1, best_edge_idx2, best_flip_idx = self.get_best_params(piece_idx1, piece_idx2)

        self.fill_index_arr(piece_idx1)
        self.change_coord(best_edge_idx1)
        self.fill_index_arr(piece_idx2)

        self.modify_pieces1(piece_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)

    def modify_pieces2(self, piece_idx1, piece_idx2, edge_idx1, edge_idx2, flip_idx):
        if piece_idx1 in self.idx_arr and piece_idx2 in self.idx_arr:
            return
        
        if piece_idx1 in self.idx_arr:
            trg_piece_idx = piece_idx2
            trg_edge_idx = edge_idx2
        elif piece_idx2 in self.idx_arr:
            trg_piece_idx = piece_idx1
            trg_edge_idx = edge_idx1
        else:
            return

        if trg_edge_idx in [1, 3]:
            if edge_idx2 == edge_idx1:
                self.pieces[trg_piece_idx] = hflip(self.pieces[trg_piece_idx])
            if flip_idx == 1:
                self.pieces[trg_piece_idx] = vflip(self.pieces[trg_piece_idx])
        else:
            if edge_idx2 == edge_idx1:
                self.pieces[trg_piece_idx] = vflip(self.pieces[trg_piece_idx])
            if flip_idx == 1:
                self.pieces[trg_piece_idx] = hflip(self.pieces[trg_piece_idx])

    def check_termination_flag(self, M, N):
        """
        `self.idx_arr`에 모든 조각들의 인덱스가 존재한다면 `True`를 반환합니다.
        """
        return (self.idx_arr != 255).sum() >= M * N

    def transform_pairs(pairs, pair_idx):
        """
        """
        print("A")
        new_pairs = deepcopy(pairs)
        for trg_pair_idx in range(pair_idx + 1, len(pairs)):
            new_pairs[trg_pair_idx + 1] = pairs[trg_pair_idx]
        new_pairs[pair_idx + 2] = pairs[pair_idx]
        return new_pairs

    def get_piece_order(self, pairs, M, N):
        pair_idx = 1
        while True:
            if self.check_termination_flag(M=M, N=N):
                break

            piece_idx1, piece_idx2 = pairs[pair_idx]
            _, best_edge_idx1, best_edge_idx2, best_flip_idx = self.get_best_params(piece_idx1, piece_idx2)

            coord1 = self.get_coord_of_idx(piece_idx1)
            coord2 = self.get_coord_of_idx(piece_idx2)
            if coord1:
                self.coord = coord1
                self.modify_pieces2(piece_idx1, piece_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
                self.change_coord(best_edge_idx1)
                self.fill_index_arr(piece_idx2)
            elif coord2:
                self.coord = deepcopy(coord2)
                self.modify_pieces2(piece_idx1, piece_idx2, best_edge_idx1, best_edge_idx2, best_flip_idx)
                self.change_coord(best_edge_idx2)
                self.fill_index_arr(piece_idx1)
            else:
                pairs = self.transform_pairs(pairs, pair_idx=pair_idx)
            pair_idx += 1
        order = list(self.idx_arr[(self.idx_arr != 255)])
        return order

    def merge_pieces_by_order(self, order, M, N):
        sub_h, sub_w, _ = self.pieces[0].shape
        merged = np.empty(
            shape=(sub_h * M, sub_w * N, 3), dtype="uint8",
        )
        for row in range(M):
            for col in range(N):
                merged[
                    row * sub_h: (row + 1) * sub_h,
                    col * sub_w: (col + 1) * sub_w,
                    :,
                ] = self.pieces[order[row * N + col]]
        return merged

    def merge(self, M, N):
        pairs = self.sort_piece_indices_pairs_by_min_dist()
        self.init_index_array(pairs=pairs, M=M, N=N)
        order = self.get_piece_order(pairs=pairs, M=M, N=N)
        merged = self.merge_pieces_by_order(order=order, M=M, N=N)
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

    def solve(self, input_dir, M, N):
        self.pieces = load_pieces(input_dir)

        if M == N:
            merged = self.merge(M=M, N=N)
        else:
            merged1 = self.merge(M=M, N=N)
            grid_dist1 = self.calc_l2_dist_along_cell_edges(merged1, M=M, N=N)
            merged2 = self.merge(M=N, N=M)
            grid_dist2 = self.calc_l2_dist_along_cell_edges(merged2, M=N, N=M)
            merged = merged1 if grid_dist1 <= grid_dist2 else merged2
        return merged

    def save(self, input_dir, M, N, save_path):
        start_time = time.time()

        solved = self.solve(input_dir=input_dir, M=M, N=N)
        save_image(solved, save_path=save_path)
        print(f"Completed solving the Jigsaw puzzle! ({self.get_elapsed_time(start_time)} elapsed.)")


def main():
    args = get_args()

    if args.M * args.N != len(list(Path(args.INPUT_DIR).glob("*.png"))):
        print("Wrong number of pieces!")
    else:
        model = JigsawPuzzleSolver()
        model.save(
            input_dir=args.INPUT_DIR, M=args.M, N=args.N, save_path=args.SAVE_PATH,
        )


if __name__ == "__main__":
    main()
