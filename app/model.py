# app/model.py

import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

# Choose the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model identifier
MODEL_NAME = "deepseek-ai/deepseek-vl2-tiny"

# Load processor and model
processor = DeepseekVLV2Processor.from_pretrained(MODEL_NAME)
model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Move model to device and set eval mode
model = model.to(device).eval()


def extract_text_from_image(image):
    """
    Extract alphanumeric characters from an input image using DeepSeek-VL2-Tiny.

    Args:
        image (PIL.Image or tensor): Input image.

    Returns:
        str: Extracted alphanumeric text.
    """
    prompt = (
        "<image> Extract the alphanumeric characters of this image, "
        "respond with the characters only without spaces between them."
    )

    # Prepare inputs for the model
    inputs = processor(images=[image], prompt=prompt, return_tensors="pt").to(device)

    # Generate model output
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs.embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            do_sample=False,
        )

    # Decode the output, skipping special tokens
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
