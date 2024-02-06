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


def transform(img):
    if random.random() < 0.5:
        img = vflip(img)
    if random.random() < 0.5:
        img = hflip(img)
    if random.random() < 0.5:
        img = rotate(img)
    return img


def get_rand_num():
    return random.randint(10 ** 9, (10 ** 10) - 1)


def load_patches(patches_dir):
    patches = dict()
    for idx, img_path in enumerate(Path(patches_dir).glob("*.png")):
        patch = load_image(img_path)
        sub_h, sub_w, _ = patch.shape
        if sub_w < sub_h:
            patch = rotate(patch)
        patches[idx] = patch
    return patches



# def merge_patches(patches, idx_arr, M, N):
#     x, y = np.where(idx_arr != 255)
#     idx_arr[x[0]: x[1] + 1, y[0]: y[1] + 1][0]
#     order

#     sub_h, sub_w, _ = patches[0].shape
#     merged = np.empty(
#         shape=(sub_h * M, sub_w * N, 3), dtype="uint8",
#     )
#     for row in range(M):
#         for col in range(N):
#             merged[
#                 row * sub_h: (row + 1) * sub_h,
#                 col * sub_w: (col + 1) * sub_w,
#                 :,
#             ] = patches[order[row * N + col]]
#     return merged


def empty_dir(trg_dir):
    try:
        path = Path(trg_dir)
        for item in path.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        
        print(f"Emptied the directory'{trg_dir}'!")
    except Exception as e:
        print(f"Error occured while trying to empty '{trg_dir}';\n{e}")
