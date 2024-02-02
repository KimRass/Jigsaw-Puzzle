import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
import requests
from pathlib import Path
import pandas as pd
from scipy.sparse import coo_matrix
from skimage.feature import peak_local_max
from skimage.morphology import local_maxima
from skimage.segmentation import watershed
from moviepy.video.io.bindings import mplfig_to_npimage
from io import BytesIO


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
    cv2.imwrite(
        filename=str(save_path), img=img[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100],
    )
