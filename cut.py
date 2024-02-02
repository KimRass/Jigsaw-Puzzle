from PIL import Image
import albumentations as A
import random
from pathlib import Path
import argparse
import numpy as np

from utils import load_image, show_image, save_image


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


def vflip(img):
    return img[::-1, ...]


def hflip(img):
    return img[:, ::-1, :]


def rotate(img):
    temp = np.transpose(img, axes=(1, 0, 2))
    temp = hflip(temp)
    return temp


def transform(img):
    # if random.random() < 0.5:
    #     img = A.rotate(img, angle=-90)
    transformer = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-90, -90), rotate_method="ellipse", p=0.5),
            # A.Rotate(limit=(-90, -90), p=1),
        ]
    )
    return transformer(image=img)["image"]
show_image(img)
show_image(rotate(img))

show_image(np.transpose(img, axes=(1, 0, 2)))

def get_rand_num():
    return random.randint(10 ** 9, (10 ** 10) - 1)


def main():
    args = get_args()

    img = load_image("/Users/jongbeomkim/Desktop/workspace/PGGAN/generated_images/512×512_2.jpg")

    n_row_splits = 4
    n_col_splits = 3
    h, w, _ = img.shape

    sub_h = h // n_row_splits
    sub_w = w // n_col_splits

    save_dir = Path("/Users/jongbeomkim/Documents/dmeta")
    for row in range(n_row_splits):
        for col in range(n_col_splits):
            sub_img = img[
                row * sub_h: (row + 1) * sub_h,
                col * sub_w: (col + 1) * sub_w,
                :,
            ]
            show_image(sub_img)
            sub_img = transform(sub_img)
            show_image(sub_img)
            rand_num = get_rand_num()
            # save_image(sub_img, save_path=save_dir/f"{rand_num}.png")


if __name__ == "__main__":
    main()
