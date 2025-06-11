import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from model import extract_text

app = FastAPI()
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def ping():
    return {"text": "Hello, I am awake"}


@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    # Ensure it's an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Read the image
    contents = await file.read()
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(image_path, "wb") as f:
        f.write(contents)

    result = extract_text(image_path)
    return {"text": result.replace("<｜end▁of▁sentence｜", "")}
