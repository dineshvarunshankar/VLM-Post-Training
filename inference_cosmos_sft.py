import json
import os
import torch
from unsloth import FastModel

model_path = "outputs/cosmos_reason2/lora_20260507_124402/fused_model_weights"
test_file = "test/test.jsonl"
output_file = "test/cosmos_predictions.json"
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
        fullgraph = False, # only if multiple GPUs are used
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
            "sft_cot": response,
            "gt_cot": sample.get('cot', ''),
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Done! Results saved to: {output_file}")
