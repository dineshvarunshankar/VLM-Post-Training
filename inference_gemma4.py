from unsloth.chat_templates import get_chat_template


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    dtype = None, # None for auto detection
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # full finetuning 
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
    device_map = "balanced" # Use 2x Tesla T4s 
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4-thinking",
)

messages = [{
    
    
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    use_cache = True,
    # Recommended Gemma-4 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)