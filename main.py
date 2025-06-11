# app/main.py
from app.model import extract_text_from_image
from app.utils import load_image_from_bytes

for i in range(1, 9):
    with open(f"/app/tests/{i}.jpg", "rb") as f:
        img = load_image_from_bytes(f.read())
    result = extract_text_from_image(img)
    print(f"result of {i} is {result}")
    with open("results.csv", "a") as f:
        f.write(f"{i}.jpg, {result}")
