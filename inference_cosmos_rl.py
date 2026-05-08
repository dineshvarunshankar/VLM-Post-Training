import json
import os
from pathlib import Path

import torch
from unsloth import FastModel

# Point this to your RL (GRPO) checkpoint.
model_path = "outputs/cosmos_grpo/REPLACE_WITH_RUN/fused_model_weights"
test_file = "test/test.jsonl"
predictions_file = "test/cosmos_predictions.json"
max_new_tokens = 12288


def load_model(model_path):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        dtype=torch.bfloat16,  # NVIDIA: "We have only tested doing inference with BF16 precision."
        max_seq_length=16384,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="auto",
        fullgraph=False,  # only if multiple GPUs are used
    )
    return model, tokenizer


def run_inference(model, tokenizer, image_path, question):
    reasoning_prompt = (
        f"{question} Answer the question using the following exact format:\n\n"
        "<think>\n[Provide the subject bounding box as [x1, y1, x2, y2], then reason to a yes/no conclusion]\n</think>\n\n"
        "<answer>\n[Your final answer]\n</answer>"
        "\n\nThe final answer must be exactly yes or no."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": reasoning_prompt},
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
            temperature=0.7,
            top_p=0.9,
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

    print(f"Loading RL model from: {model_path}")
    model, tokenizer = load_model(model_path)

    print(f"Running RL inference on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {os.path.basename(sample['image'])}")
        rows[i]["rl_cot"] = run_inference(
            model, tokenizer, sample["image"], sample["question"]
        )

    outp = Path(predictions_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Done! Wrote {len(rows)} rows with rl_cot to: {predictions_file}")
