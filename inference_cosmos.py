"""
Inference script for finetuned Cosmos-Reason2-8B model.

Supports two modes:
  1. LoRA adapter: loads base model + adapter weights (slower startup, smaller disk usage)
  2. Merged model: loads the fused model weights (faster inference, larger disk usage)

Usage:
  python inference_cosmos.py --image_path path/to/image.jpg --question "Is it safe to turn right?"
  python inference_cosmos.py --image_path path/to/image.jpg --question "Describe the scene" --model_path outputs/cosmos_reason2/lora_YYYYMMDD_HHMMSS/fused_model_weights
"""

import argparse
import torch
from PIL import Image
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def load_model(model_path, load_merged=False):
    """
    Load the Cosmos-Reason2 model for inference.

    Args:
        model_path: Path to either:
            - LoRA adapter directory (e.g., outputs/cosmos_reason2/lora_.../adapter_weights)
            - Merged model directory (e.g., outputs/cosmos_reason2/lora_.../fused_model_weights)
        load_merged: If True, loads a fully merged model. If False, loads base + LoRA adapter.
    """
    if load_merged:
        # Load the fully merged model (no adapter needed)
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            dtype=torch.bfloat16,
            max_seq_length=8192,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            device_map="auto",
        )
    else:
        # Load base model + LoRA adapter
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,  # Points to LoRA adapter dir (it stores base model ref)
            dtype=torch.bfloat16,
            max_seq_length=8192,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            device_map="auto",
        )

    # Inject the ChatML template (Cosmos is Qwen3-VL based)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    return model, tokenizer


def run_inference(model, tokenizer, image_path, question, max_new_tokens=4096):
    """
    Run inference on a single image with a question.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer with chat template.
        image_path: Path to the input image (jpg/png).
        question: The question to ask about the image.
        max_new_tokens: Maximum number of tokens to generate (4096 recommended for long CoT).

    Returns:
        str: The model's response (including <think>/<answer> blocks).
    """
    # Construct the reasoning prompt per NVIDIA's model card
    reasoning_prompt = (
        f"{question} Answer the question using the following format:\n\n"
        "<think>\nYour reasoning.\n</think>\n\n"
        "Write your final answer immediately after the </think> tag."
    )

    # Build the message in Unsloth's expected format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": reasoning_prompt},
            ],
        }
    ]

    # Tokenize with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Adds <|im_start|>assistant\n at the end
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.6,       # Lower temp for more deterministic reasoning
            top_p=0.9,
            do_sample=True,
        )

    # Trim the input tokens from the output to get only the generated response
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]

    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Cosmos-Reason2-8B Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/cosmos_reason2/lora_latest/adapter_weights",
        help="Path to LoRA adapter or merged model directory",
    )
    parser.add_argument("--merged", action="store_true", help="Set if loading a merged (fused) model instead of LoRA adapter")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max tokens to generate (4096 recommended for CoT)")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, load_merged=args.merged)

    print(f"Running inference on: {args.image_path}")
    print(f"Question: {args.question}")
    print("-" * 60)

    response = run_inference(
        model, tokenizer,
        image_path=args.image_path,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )

    print(response)


if __name__ == "__main__":
    main()
