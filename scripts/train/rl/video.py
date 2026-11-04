"""Preserve native-video processor fields when scoring RL completions."""

import torch


def prepare_video_prompt(collator, examples):
    """Use native SFT media processing for RL and evaluation prompts."""
    if not examples:
        raise ValueError('A media prompt batch cannot be empty')
    prompt = collator.prompt_batch(examples)
    media_types = {("video" if row.get("media_type") is None else row["media_type"]) for row in examples}
    if len(media_types) != 1 or not media_types <= {"image", "video"}:
        raise ValueError("Prompt batches require one supported media_type")
    media = next(iter(media_types))
    field = "pixel_values" if media == "image" else "pixel_values_videos"
    if field not in prompt or prompt[field].numel() == 0:
        raise ValueError(f"Processor did not produce native-{media} tensors")
    if prompt['input_ids'].shape[1] >= collator.max_length:
        raise ValueError('Media prompt leaves no context space for a completion')
    return prompt


def completion_inputs(prompt, completion_ids, completion_mask, *, expected_prompt_ids, max_length):
    """Attach sampled tokens without regenerating or discarding visual inputs."""
    video = prompt.get('pixel_values_videos')
    if video is None or video.numel() == 0:
        raise ValueError('RL forward pass requires native-video tensors')
    ids, mask = prompt['input_ids'], prompt['attention_mask']
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError('Prompt token IDs and attention masks must have matching batch shapes')
    if expected_prompt_ids.shape != ids.shape or not torch.equal(expected_prompt_ids.to(ids.device),ids):
        raise ValueError('Rollout prompt tokens differ from training processor tokens')
    if completion_ids.ndim != 2 or completion_ids.shape != completion_mask.shape or completion_ids.shape[0] != ids.shape[0]:
        raise ValueError('Completion IDs and masks must match the prompt batch')
    if completion_ids.shape[1] == 0:
        raise ValueError('A completion must contain at least one token')
    if not torch.all((completion_mask == 0) | (completion_mask == 1)):
        raise ValueError('Completion mask must be binary')
    if ids.shape[1] + completion_ids.shape[1] > max_length:
        raise ValueError('Prompt and completion exceed the model context budget')
    if 'position_ids' in prompt:
        raise ValueError('Explicit text position IDs require a model-specific continuation adapter')
    result = dict(prompt)
    completion_ids = completion_ids.to(device=ids.device,dtype=ids.dtype)
    completion_mask = completion_mask.to(device=mask.device,dtype=mask.dtype)
    result['input_ids'] = torch.cat((ids,completion_ids),dim=1)
    result['attention_mask'] = torch.cat((mask,completion_mask),dim=1)
    for key in ('token_type_ids','mm_token_type_ids'):
        if key in prompt:
            field = prompt[key]
            if field.shape != ids.shape:
                raise ValueError(f'{key} must align with prompt tokens')
            result[key] = torch.cat((field,field.new_zeros(completion_ids.shape)),dim=1)
    labels = result['input_ids'].masked_fill(result['attention_mask']==0,-100)
    labels[:,:ids.shape[1]] = -100
    result['labels'] = labels
    return result


def video_log_probs(model, inputs, completion_length):
    """Use TRL's fused forward contract, retaining gradient flow through video modules."""
    if 'pixel_values_videos' not in inputs:
        raise ValueError('Video fields are missing from policy forward inputs')
    if type(completion_length) is not int or not 0 < completion_length < inputs['input_ids'].shape[1]:
        raise ValueError('Invalid completion length')
    output = model(**inputs,fused_lm_head=True)
    if getattr(output,'log_probs',None) is None or getattr(output,'entropy',None) is None:
        raise ValueError('Model must be prepared with the supported TRL fused forward implementation')
    expected = (inputs['input_ids'].shape[0],completion_length)
    logps = output.log_probs[:,-completion_length:]
    entropy = output.entropy[:,-completion_length:]
    if logps.shape != expected or entropy.shape != expected:
        raise ValueError('Policy log probabilities do not align with completion tokens')
    return logps,entropy


def stack_video_prompts(prompts, *, pad_token_id, prompt_width, device):
    """Batch cached single-video processor outputs without decoding videos again."""
    if not prompts or any(p['input_ids'].shape[0] != 1 for p in prompts):
        raise ValueError('Expected one video example per cached prompt')
    keys = set(prompts[0])
    if any(set(p) != keys for p in prompts):
        raise ValueError('Video processor fields differ within one model batch')
    token_fields = {'input_ids','attention_mask','token_type_ids','mm_token_type_ids'}
    result = {}
    for key in keys:
        values = [p[key] for p in prompts]
        if key in token_fields:
            if any(v.ndim != 2 or v.shape[1] > prompt_width for v in values):
                raise ValueError('Cached prompt exceeds rollout padding width')
            fill = pad_token_id if key == 'input_ids' else 0
            values = [torch.nn.functional.pad(v,(prompt_width-v.shape[1],0),value=fill) for v in values]
        if all(isinstance(v,torch.Tensor) for v in values):
            try:
                result[key] = torch.cat(values,dim=0).to(device)
            except RuntimeError as error:
                raise ValueError(f'Incompatible cached video tensor shapes: {key}') from error
        elif key == 'num_frames_per_video' and all(isinstance(v,list) for v in values):
            result[key] = [item for value in values for item in value]
        else:
            raise ValueError(f'Unsupported cached processor field: {key}')
    return result
