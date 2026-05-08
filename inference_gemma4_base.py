"""
Base Gemma-4-31B-it (HF, no LoRA). Reads predictions JSON in place and adds each row's `base` field back to the same file.
"""

import json
import os
from pathlib import Path

import torch
from unsloth import FastModel

model_name = "unsloth/gemma-4-31B-it"
test_file = "test/test.jsonl"
predictions_file = "test/gemma4_predictions.json"
max_new_tokens = 8192


def load_model(model_name):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=torch.bfloat16,
        max_seq_length=8192,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="balanced",
    )
    return model, tokenizer


def run_inference(model, tokenizer, image_path, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
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
    with open(predictions_file, encoding="utf-8") as f:
        rows = json.load(f)
    with open(test_file, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == len(samples), f"{len(rows)} rows in {predictions_file} vs {len(samples)} jsonl lines"

    print(f"Loading base model from: {model_name}")
    model, tokenizer = load_model(model_name)

    print(f"Running base inference on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {os.path.basename(sample['image'])}")
        rows[i]["base"] = run_inference(
            model, tokenizer, sample["image"], sample["question"]
        )

    outp = Path(predictions_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Done! Wrote {len(rows)} rows with base to: {predictions_file}")
