import os

from pyrogram import Client, filters

from model import extract_text

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

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
    await message.reply(result.replace("<｜end▁of▁sentence｜", ""))


app.run()  # Automatically start() and idle()
