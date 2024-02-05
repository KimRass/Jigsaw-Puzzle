import albumentations as A
from pathlib import Path
import argparse

from utils import (
    load_image,
    show_image,
    save_image,
    transform,
    get_rand_num,
    empty_dir,
)


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

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

    img = load_image(args.IMG_PATH)

    empty_dir(args.SAVE_DIR)

    h, w, _ = img.shape
    sub_h = h // args.M
    sub_w = w // args.N
    for row in range(args.M):
        for col in range(args.N):
            sub_img = img[
                row * sub_h: (row + 1) * sub_h,
                col * sub_w: (col + 1) * sub_w,
                :,
            ]
            sub_img = transform(sub_img)
            rand_num = get_rand_num()
            save_image(sub_img, save_path=Path(args.SAVE_DIR)/f"{rand_num}.png")
    print(f"Completed cutting the image into {args.M} x {args.N} patches!")


if __name__ == "__main__":
    main()
