import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
    model_path
)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


PROMPT = """
You are an advanced visual recognition model. Your task is to analyze the image and extract the alphanumeric character sequence it contains. The sequence may include:
Uppercase letters (A–Z)
Lowercase letters (a–z)
Digits (0–9)
Characters may appear in various handwriting styles or machine-printed fonts, and they may vary in size and clarity. Carefully trace all visible lines — every mark may be part of a character.
Important Instructions:
Distinguish clearly between similar characters (e.g., ‘O’ vs ‘0’, ‘I’ vs ‘1’, ‘l’ vs ‘1’, etc.).
Prioritize letters over digits when uncertain — assume a character is a letter unless there's strong visual evidence it is a number.
Each character must be counted only once. No character should be interpreted as two.
Do not include any spaces, punctuation, or special characters in the output — return only the continuous alphanumeric sequence.
If the image contains a mathematical problem, extract the complete expression, solve it, and return only the final answer (not the expression itself).
Output format:
Return only the cleaned and uninterrupted alphanumeric sequence or, if it's a math problem, return the numeric result
"""


## single image conversation example
## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
def extract_text(img_path):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image> {PROMPT}",
            "images": [img_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    return answer


if __name__ == "__main__":
    for i in range(1, 9):
        extract_text(f"/app/tests/{i}.jpg")
