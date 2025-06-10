# app/utils.py
from io import BytesIO

from PIL import Image


def load_image_from_bytes(image_bytes: bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")
