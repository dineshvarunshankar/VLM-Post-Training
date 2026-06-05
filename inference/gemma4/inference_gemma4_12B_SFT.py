"""
Inference script for finetuned Gemma-4-31B.
Reads test.jsonl, runs inference, saves predictions to JSON.
"""

import json
import os

import torch
from unsloth import FastModel

model_path = "outputs/gemma_4/lora_20260507_135059/checkpoint-50"
test_split = "your_test_split"
test_file = f"data/test/{test_split}/exports/test.jsonl"
output_file = f"data/test/{test_split}/exports/gemma4_predictions.json"
max_new_tokens = 8192


def load_model(model_path):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        dtype=torch.bfloat16,
        max_seq_length=16384,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="auto", 
    )
    return model, tokenizer


def run_inference(model, tokenizer, image_path, question):
    prompt_instruction = (
        f"{question} First, identify and output the bounding box coordinates of the subject in the green box. Then, provide your step-by-step reasoning before answering."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_instruction},
        ],
    }]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device)


    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]

    return tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


if __name__ == "__main__":
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path)

    with open(test_file, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Running inference on {len(samples)} samples...")
    results = []

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {os.path.basename(sample['image'])}")
        response = run_inference(model, tokenizer, sample['image'], sample['question'])

        results.append({
            "image": sample['image'],
            "question": sample['question'],
            "gt_answer": sample.get('answer', ''),
            "prediction": response,
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Done! Results saved to: {output_file}")