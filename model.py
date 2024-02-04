import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from merge_junior import load_patchs, merge, show_image


def merge(patchs, n_row_splits, n_col_splits):
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
            ] = patchs[row * n_col_splits + col]
    return merged

img_size = 64
patch_size = img_size // 2
proj = nn.Linear(patch_size ** 2 * 3, 4)
x = torch.randn(4, 3, img_size, img_size)
x = rearrange(
    x,
    pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
    p1=patch_size,
    p2=patch_size,
)
x = proj(x)
x.shape


if __name__ == "__main__":
    n_row_splits = 2
    n_col_splits = 2
    patchs = load_patchs("/Users/jongbeomkim/Documents/dmeta")
    merged = merge(
        patchs=patchs, n_row_splits=n_row_splits, n_col_splits=n_col_splits,
    )
    show_image(merged)
