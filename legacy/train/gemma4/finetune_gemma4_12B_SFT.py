import numpy as np
import unsloth
from trl import SFTTrainer, SFTConfig
from data.data_processor import DataProcessor
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
import torch
import os
from datetime import datetime
#for wandb project
os.environ["WANDB_PROJECT"] = "Triage-VLM-Post-Training"

#datetime string
datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')

#Available gemma4 models

gemma4_models = [
    # Gemma-4 instruct:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-12b-it",
    # Gemma-4 base:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
    "unsloth/gemma-4-12b",
]

#load the model (base 12b)
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-12b",
    dtype = torch.bfloat16, # None for auto detection or torch.float32, torch.float16, torch.bfloat16, etc.
    max_seq_length = 16384, # context window - prompt + image tokens + generated response
    load_in_4bit = False,  # 4 bit quantization to reduce memory (NF4) - This IS QLoRA, not LoRA
    load_in_8bit = False, # 8 bit quantization to reduce memory (Q8_0)
    load_in_16bit = True, # True 16-bit LoRA
    load_in_fp8 = False, # 8 bit quantization(FP8)
    full_finetuning = False, # full finetuning 
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
    device_map = "auto", # for equally distributed multi-GPU training. 31B bf16 requires ~2+ H100s (~62GB weights + LoRA + optimizer)
    use_gradient_checkpointing = "unsloth" # True or "unsloth" for long context. Essential for 31B model.
)

#finetune gemma 4

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!
    r = 32,           # Larger = higher accuracy, but might overfit. 32 for 31B model with rich CoT data.
    lora_alpha = 64,  # Set alpha=r for 31B models to prevent scaling up unstable gradients
    lora_dropout = 0, # Unsloth optimizes for dropout=0; low overfit risk at 1-2 epochs
    bias = "none",
    random_state = 3407, #a paper proved it was optimal - https://arxiv.org/abs/2109.08203  
    use_rslora = True,  # rank stabilized LoRA - alpha/sqrt(r), improves stability at higher ranks
    loftq_config = None, #  LoftQ - Calculates what was lost during compression and preloads it into LoRA adapter.
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# Data preparation - Gemma-4 uses <|channel>thought / <channel|> reasoning format
train_split = "100_100"
processor = DataProcessor(
    f"data/train/{train_split}/cot_annotations/sft.jsonl", model_type="gemma4"
)
ready_file = processor.process_and_save()
dataset = load_dataset("json", data_files=ready_file, split="train")

# NOTE: Do NOT override the chat template for vision models.
# Gemma-4's native tokenizer has a multimodal-aware chat template
# that handles list-format content (images + text). The override
# replaces it with a text-only version that crashes on multimodal data.
# The native template already uses <|turn>user/model tokens.

#standardize data formats - Hugging Face JSON to conversational arrays
dataset = standardize_data_formats(dataset)

#apply chat template - flatten the conversations into a single text field - Only for text finetuning
# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
#     return { "text" : texts, }

# dataset = dataset.map(formatting_prompts_func, batched = True)

#Train the model

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model,
        tokenizer,
        train_on_responses_only = True, #masks instruction token by setting it to -100
        # Gemma-4 specific masking with turn delimiters:
        instruction_part = "<|turn>user\n",
        response_part = "<|turn>model\n",
        completion_only_loss = True, # Only compute loss on assistant response
    ),
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        #dataset_text_field="text", #commented out for vision finetuning
        per_device_train_batch_size=2, # 31B model - keep at 1 for memory safety
        gradient_accumulation_steps=8, # Effective batch size = 1 * 16 = 16
        warmup_steps=10, # ~10% of total steps
        num_train_epochs=2, #2 epochs helps avoid underfitting with less data
        max_steps=-1, # -1 means use num_train_epochs instead. Set to positive int for testing.
        learning_rate=2e-4, # Extremely conservative LR for 31B model to prevent explosion
        max_grad_norm=0.5,  # Force clip exploding gradients
        logging_steps=1,
        optim="adamw_torch_fused", # Full precision fused AdamW - best for H100. Alt: "adamw_8bit" to save optimizer memory
        weight_decay=0.01, # Standard regularization
        lr_scheduler_type="cosine", # Cosine annealing - smoother decay than linear
        seed=3407,
        bf16=True, # only for A100s & H100s
        report_to="wandb",
        run_name=f"gemma_4_12b_base/lora_{datetime_string}",
        output_dir=f"outputs/gemma_4_12b_base/lora_{datetime_string}",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        remove_unused_columns = False, # by default, it looks at text field, if we have images, it would ignore it. Forcing this to false protects images.
        dataset_text_field = "", #stops looking for text field since we have images.
        dataset_kwargs = {"skip_prepare_dataset": True}, # tells HF to skip data processing since unsloth already processed it.
        max_length = 16384, #HF leaves room for unsloth to add tokens. Should match "max_seq_length" in the model.from_pretrained()
    )
)

# #Train on assistant outputs and ignore the loss on the user's inputs
# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="<|turn>user\n",
#     response_part="<|turn>model\n",
# )

#memory stats - initial

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#TRAINING
trainer_stats = trainer.train()

#memory stats - final
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#save the model
model.save_pretrained(f"outputs/gemma_4_12b_base/lora_{datetime_string}/adapter_weights")
tokenizer.save_pretrained(f"outputs/gemma_4_12b_base/lora_{datetime_string}/tokenizer_weights")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving

#saving to float16 for VLLM
model.save_pretrained_merged(f"outputs/gemma_4_12b_base/lora_{datetime_string}/fused_model_weights", tokenizer)

#saving to GGUF/llama.cpp conversion
#model.save_pretrained_gguf(f"outputs/gemma_4_12b_base/lora_{datetime_string}/fused_model_weights", tokenizer, quantization_method = "Q8_0") # Q8_0, BF16, F16 supported