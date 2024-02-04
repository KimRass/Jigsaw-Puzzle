from PIL import Image
import albumentations as A
import random
from pathlib import Path
import argparse
import numpy as np

from utils import (
    load_image,
    show_image,
    save_image,
    transform,
    get_rand_num,
)


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


def main():
    args = get_args()

    img = load_image("/Users/jongbeomkim/Desktop/workspace/Gatys-et-al.-2016/examples/content_images/content_image1.jpg")

    n_row_splits = 3
    n_col_splits = 3
    h, w, _ = img.shape

    sub_h = h // n_row_splits
    sub_w = w // n_col_splits

    save_dir = Path("/Users/jongbeomkim/Documents/dmeta/3by3")
    for row in range(n_row_splits):
        for col in range(n_col_splits):
            sub_img = img[
                row * sub_h: (row + 1) * sub_h,
                col * sub_w: (col + 1) * sub_w,
                :,
            ]
            sub_img = transform(sub_img)
            rand_num = get_rand_num()
            save_image(sub_img, save_path=save_dir/f"{rand_num}.png")


if __name__ == "__main__":
    main()
