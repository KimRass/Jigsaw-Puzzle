from PIL import Image
import albumentations as A
import random
from pathlib import Path
import argparse

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


def get_rand_num():
    return random.randint(10 ** 9, (10 ** 10) - 1)


def main():
    trg_dir = Path("/Users/jongbeomkim/Documents/dmeta")
    sub_imgs = list()
    for img_path in trg_dir.glob("*.png"):
        sub_img = load_image(img_path)
        sub_imgs.append(sub_img)
    # show_image(sub_img)
    