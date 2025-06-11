import os

from pyrogram import Client, filters

from model import extract_text

print("model imported")

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

app = Client("myregbot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
print("Bot created")


@app.on_message(filters.private & filters.user("@Sd_Ammar"))
async def ocr(client, message):
    if not message.photo:
        await message.reply("no photo", quote=True)
        return

    # Read the image
    image_path = await client.download_media(message.photo.file_id)
    result = (
        extract_text(image_path).replace("<｜end▁of▁sentence｜>", "").replace(" ", "")
    )
    print(result)
    await message.reply(result, quote=True)


app.run()  # Automatically start() and idle()
