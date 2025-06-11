# app/model.py
import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "deepseek-ai/deepseek-vl2-tiny"

processor = DeepseekVLV2Processor.from_pretrained(MODEL_NAME)
model = DeepseekVLV2ForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.to(device).eval()


def extract_text_from_image(image):
    prompt = "<ImageHere> Extract the alphanumeric characters of this image, respond with the characters only without spaces between them."
    inputs = processor(images=[image], prompt=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs.embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            do_sample=False,
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
