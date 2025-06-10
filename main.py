# app/main.py
from app.model import extract_text_from_image
from app.utils import load_image_from_bytes

for i in range(1, 9):
    with open(f"/tests/{i}.jpg", "rb") as f:
        img = load_image_from_bytes(f)
    result = extract_text_from_image(img)
    print(f"result of {i} is {result}")
