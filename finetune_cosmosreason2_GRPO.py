import os
from datetime import datetime
import torch
from datasets import load_dataset

from unsloth import FastModel

from unsloth.chat_templates import standardize_data_formats
from trl import GRPOConfig, GRPOTrainer

from data_processor import DataProcessor
from reward_functions import (
    FormatReward,
    AnswerCorrectnessReward,
    BboxAccuracyReward,
    ConsistencyReward,
    RewardAggregator
)

# For wandb project
os.environ["WANDB_PROJECT"] = "Triage-VLM-Post-Training"

# Datetime string
datetime_string = datetime.now().strftime('%Y%m%d_%H%M%S')

print("Loading model...")
model, tokenizer = FastModel.from_pretrained(
    model_name="nvidia/Cosmos-Reason2-8B",
    dtype=torch.bfloat16,
    max_seq_length=16384,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    load_in_fp8=False,
    full_finetuning=False,
    device_map="auto",
    use_gradient_checkpointing="unsloth",
    # MUST BE FALSE for thermal images since we must train vision layers
    fast_inference=False 
)

print("Setting up PEFT model...")
model = FastModel.get_peft_model(
    model,
    # MUST BE TRUE to adapt to thermal images
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=True,
)

print("Preparing GRPO dataset...")
processor = DataProcessor("outputs/dataset/sft.jsonl", model_type="cosmos")
ready_file = processor.process_for_grpo()
dataset = load_dataset("json", data_files=ready_file, split="train")
dataset = standardize_data_formats(dataset)

print("Setting up Reward System...")
reward_system = RewardAggregator(
    reward_functions=[
        FormatReward(),
        AnswerCorrectnessReward(),
        BboxAccuracyReward(),
        ConsistencyReward()
    ],
    # Correctness dominates, while bbox/format/consistency still provide GRPO signal.
    weights=[1.0, 6.0, 2.0, 1.0],
    normalize=True,
)

print("Configuring GRPO Trainer...")
training_args = GRPOConfig(
    output_dir=f"outputs/cosmos_grpo/{datetime_string}",
    num_train_epochs=2,
    num_generations=16,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    logging_steps=1,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    report_to="wandb",
    run_name=f"cosmos_grpo_{datetime_string}",
    
    # GRPO specific
    use_vllm=False,  # Cannot use vLLM with vision LoRA
    max_prompt_length=8192,
    max_completion_length=8192,
    beta=0.001,
    loss_type="bnpo",
    temperature=1.0,
    top_p=1.0,
    log_completions=True,
    
    # Optimizer
    optim="adamw_torch_fused",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    max_grad_norm=1.0,
    
    # Memory optimization
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_system,
)

# Memory stats - initial
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print("Starting training...")
trainer_stats = trainer.train()

# Memory stats - final
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

# Save the model
print("Saving model...")
model.save_pretrained(f"outputs/cosmos_grpo/{datetime_string}/adapter_weights")
tokenizer.save_pretrained(f"outputs/cosmos_grpo/{datetime_string}/tokenizer_weights")

# Saving to float16 for VLLM
model.save_pretrained_merged(f"outputs/cosmos_grpo/{datetime_string}/fused_model_weights", tokenizer)
print("Finished!")
