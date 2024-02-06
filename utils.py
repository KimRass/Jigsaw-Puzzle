import numpy as np
import requests
import cv2
from PIL import Image
from pathlib import Path
from io import BytesIO
import random


def load_image(url_or_path):
    url_or_path = str(url_or_path)
    if url_or_path[: 4] == "http":
        response = requests.get(url_or_path)
        url_or_path = BytesIO(response.content)
    image = Image.open(url_or_path).convert("RGB")
    img = np.array(image)
    return img


def to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img, mode="RGB")
    return img


def show_image(img):
    to_pil(img).show()


def save_image(img, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        filename=str(save_path), img=img[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100],
    )


def vflip(img):
    return img[:: -1, ...]


def hflip(img):
    return img[:, :: -1, :]


def rotate(img):
    temp = np.transpose(img, axes=(1, 0, 2))
    temp = hflip(temp)
    return temp


def load_patches(patches_dir):
    patches = dict()
    for idx, img_path in enumerate(Path(patches_dir).glob("*.png")):
        patch = load_image(img_path)
        sub_h, sub_w, _ = patch.shape
        if sub_w < sub_h:
            patch = rotate(patch)
        patches[idx] = patch
    return patches
