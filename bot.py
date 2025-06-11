from pyrogram import Client, filters

from model import extract_text

API_ID = "23678585"
API_HASH = "0ef7b2d89db102e3347e0b73f6d4ab6e"
BOT_TOKEN = "7570245938:AAFgoNDSOc-OjHj7IuzoEnQIdMdJRnr-ZTs"

app = Client("myregbot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)


@app.on_message(filters.private & filters.user("@Sd_Ammar"))
async def ocr(client, message):
    if not message.photo:
        await message.reply("no photo")
        return

    # Read the image
    image_path = await client.download_media(message.photo.file_id)
    await message.reply(image_path)
    result = extract_text(image_path)
    return {"text": result.replace("<｜end▁of▁sentence｜", "")}


app.run()  # Automatically start() and idle()
