import numpy as np
import unsloth
from trl import SFTTrainer, SFTConfig
from data_processor import DataProcessor
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset

#Available gemma4 models

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
]

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    dtype = None, # None for auto detection
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # full finetuning 
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
    device_map = "balanced" # Use 2x Tesla T4s
)

#finetune gemma 4

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!
    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

#Data preparation
processor = DataProcessor("exports/outputs/sft.jsonl")
ready_file = processor.process_and_save_gemma4()
dataset = load_dataset("json", data_files=ready_file, split="train")

#get gemmma4 template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4-thinking",
)

#Dataset preparation

dataset = load_dataset("json", data_files="exports/outputs/sft.json")

#standardize data formats
dataset = standardize_data_formats(dataset)

#apply chat template
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

#train the model
trainer = Trainer(
    model = model,
)

#Train the model

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )
)

#Train on assistant outputs and ignore the loss on the user's inputs
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

#memory stats - initial

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")

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
model.save_pretrained("gemma_4_lora")
tokenizer.save_pretrained("gemma_4_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving

#saving to float16 for VLLM
#model.save_pretrained_merged("gemma-4-finetune", tokenizer)

#saving to GGUF/llama.cpp conversion
model.save_pretrained_gguf("gemma_4_finetune", tokenizer, quantization_method = "Q8_0") # Q8_0, BF16, F16 supported