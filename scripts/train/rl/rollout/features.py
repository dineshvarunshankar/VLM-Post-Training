"""Encode single-image/video HF inputs for patched vLLM 0.30.0.

Supports native Qwen3.5 and Gemma4 images and unpruned video inputs. The rollout server must use
matching weights, model configuration, and the video placeholder-mask patch.
"""

import hashlib
from importlib.metadata import version

import torch


_TEXT_FIELDS = {'input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _video_layout(packet, video_token_id, start_token_id, end_token_id):
    ids, attention = packet['input_ids'], packet['attention_mask']
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != attention.shape:
        raise ValueError('Expected one video prompt with an aligned attention mask')
    if not torch.all((attention == 0) | (attention == 1)):
        raise ValueError('Prompt attention mask must be binary')
    tokens = ids[0][attention[0].bool()].tolist()
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('Prompt IDs must be nonnegative integers')
    runs = []
    for index, token in enumerate(tokens):
        if token != video_token_id:
            continue
        if runs and sum(runs[-1]) == index:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    if not runs:
        raise ValueError('Prompt has no visual video tokens')
    for start, length in runs:
        if (start == 0 or start + length >= len(tokens)
                or tokens[start - 1] != start_token_id
                or tokens[start + length] != end_token_id):
            raise ValueError('Video tokens lack the expected frame delimiters')
    # Include the first start marker: Qwen's M-RoPE scans for it from this offset.
    offset = runs[0][0] - 1
    end = sum(runs[-1]) + 1
    placeholder = {
        'offset': offset,
        'length': end - offset,
        'is_embed': [token == video_token_id for token in tokens[offset:end]],
    }
    return tokens, [length for _, length in runs], placeholder


def encode_video_features(packet, *, model_type, video_token_id,
                          vision_start_token_id, vision_end_token_id,
                          policy_version, spatial_merge_size=None,
                          pooling_kernel_size=None, image_token_id=None):
    """Encode native images or videos; retain this entrypoint for existing callers.

    policy_version must identify the run and synchronized weight update. Changing
    it invalidates encoder cache entries when vision adapters have changed.
    """
    if 'pixel_values' in packet:
        return encode_image_features(
            packet, model_type=model_type, image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id, policy_version=policy_version,
            spatial_merge_size=spatial_merge_size, pooling_kernel_size=pooling_kernel_size,
        )
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError('A nonempty synchronized policy version is required')
    tokens, counts, placeholder = _video_layout(
        packet, video_token_id, vision_start_token_id, vision_end_token_id
    )
    if model_type == 'qwen3_5':
        video_fields = {'pixel_values_videos', 'video_grid_thw'}
    elif model_type in ('gemma4', 'gemma4_unified'):
        video_fields = {'pixel_values_videos', 'video_position_ids', 'num_frames_per_video'}
    else:
        raise ValueError(f'Unsupported video transport architecture: {model_type}')
    if not video_fields <= packet.keys() or set(packet) - (_TEXT_FIELDS | video_fields):
        raise ValueError('Missing video fields or unexpected processor inputs')
    pixels = packet['pixel_values_videos'].detach().cpu().contiguous()
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError('Video pixels must be finite floating-point values')

    # Import only at the transport boundary; SFT does not require vLLM installed.
    if version('vllm') != '0.30.0':
        raise ValueError('Video feature transport is verified against patched vLLM 0.30.0 only')
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import PlaceholderRangeInfo
    from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import encode_mm_kwargs_item
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems

    if 'is_embed' not in PlaceholderRangeInfo.model_fields:
        raise ValueError('vLLM video placeholder-mask patch is required')
    field = MultiModalFieldConfig
    if model_type == 'qwen3_5':
        merge = _positive_int(spatial_merge_size, 'spatial_merge_size')
        grid = packet['video_grid_thw'].detach().cpu().contiguous()
        if (grid.shape != (1, 3) or grid.is_floating_point() or grid.dtype == torch.bool
                or not torch.all(grid > 0)):
            raise ValueError('Expected one positive integer video grid')
        temporal, height, width = grid[0].tolist()
        if (height % merge or width % merge or pixels.ndim != 2
                or pixels.shape[0] != temporal * height * width
                or counts != [height * width // merge**2] * temporal):
            raise ValueError('Qwen video grid, pixel rows, and visual tokens disagree')
        values = {'pixel_values_videos': pixels, 'video_grid_thw': grid}
        fields = {
            'pixel_values_videos': field.flat_from_sizes('video', grid.prod(-1)),
            'video_grid_thw': field.batched('video', keep_on_cpu=True),
        }
    else:
        positions = packet['video_position_ids'].detach().cpu().contiguous()
        frames = torch.as_tensor(packet['num_frames_per_video']).detach().cpu()
        if (frames.shape != (1,) or frames.is_floating_point() or frames.dtype == torch.bool
                or pixels.ndim != 3 or int(frames[0]) != pixels.shape[0]
                or positions.shape != (*pixels.shape[:2], 2)
                or positions.is_floating_point() or positions.dtype == torch.bool):
            raise ValueError('Gemma frame counts, positions, and pixel shapes disagree')
        valid = (positions >= 0).all(-1)
        padding = (positions == -1).all(-1)
        if not torch.all(valid | padding):
            raise ValueError('Gemma patch coordinates contain malformed padding')
        pool = 1 if model_type == 'gemma4_unified' else _positive_int(
            pooling_kernel_size, 'pooling_kernel_size')**2
        valid_counts = valid.sum(-1)
        if torch.any(valid_counts % pool) or counts != (valid_counts // pool).tolist():
            raise ValueError('Gemma patch positions and visual tokens disagree')
        values = {'pixel_values_videos': pixels, 'pixel_position_ids_videos': positions,
                  'video_frame_counts': frames}
        fields = {
            'pixel_values_videos': field.flat_from_sizes('video', frames),
            'pixel_position_ids_videos': field.flat_from_sizes('video', frames),
            'video_frame_counts': field.batched('video', keep_on_cpu=True),
        }
    item = MultiModalKwargsItems.from_hf_inputs(values, fields)['video'][0]
    encoded = encode_mm_kwargs_item(item)
    digest = hashlib.sha256()
    for value in (policy_version, model_type, encoded):
        digest.update(value.encode('utf-8'))
        digest.update(b'\0')
    return {
        'token_ids': tokens,
        'features': {
            'mm_hashes': {'video': [digest.hexdigest()]},
            'mm_placeholders': {'video': [placeholder]},
            'kwargs_data': {'video': [encoded]},
        },
    }


def encode_image_features(packet, *, model_type, image_token_id,
                          vision_start_token_id, vision_end_token_id,
                          policy_version, spatial_merge_size=None,
                          pooling_kernel_size=None):
    """Serialize one native image using vLLM's image fields and token offsets."""
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError('A nonempty synchronized policy version is required')
    _positive_int(image_token_id, 'image_token_id')
    tokens, counts, delimited = _video_layout(
        packet, image_token_id, vision_start_token_id, vision_end_token_id)
    if len(counts) != 1:
        raise ValueError('Expected one contiguous native image placeholder')
    # Unlike Qwen video M-RoPE, image M-RoPE starts at the first image token.
    # Delimiters stay text and must not shift the image positional grid.
    placeholder = {'offset': delimited['offset'] + 1, 'length': counts[0],
                   'is_embed': [True] * counts[0]}
    if model_type == 'qwen3_5':
        image_fields = {'pixel_values', 'image_grid_thw'}
    elif model_type in ('gemma4', 'gemma4_unified'):
        image_fields = {'pixel_values', 'image_position_ids'}
    else:
        raise ValueError(f'Unsupported image transport architecture: {model_type}')
    if not image_fields <= packet.keys() or set(packet) - (_TEXT_FIELDS | image_fields):
        raise ValueError('Missing image fields or unexpected processor inputs')
    pixels = packet['pixel_values'].detach().cpu().contiguous()
    if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
        raise ValueError('Image pixels must be finite floating-point values')

    if version('vllm') != '0.30.0':
        raise ValueError('Image feature transport is verified against patched vLLM 0.30.0 only')
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import PlaceholderRangeInfo
    from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import encode_mm_kwargs_item
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
    if 'is_embed' not in PlaceholderRangeInfo.model_fields:
        raise ValueError('vLLM placeholder-mask patch is required')
    field = MultiModalFieldConfig
    if model_type == 'qwen3_5':
        merge = _positive_int(spatial_merge_size, 'spatial_merge_size')
        grid = packet['image_grid_thw'].detach().cpu().contiguous()
        if (grid.shape != (1, 3) or grid.is_floating_point() or grid.dtype == torch.bool
                or not torch.all(grid > 0) or int(grid[0, 0]) != 1):
            raise ValueError('Expected one positive integer image grid with temporal dimension 1')
        _, height, width = grid[0].tolist()
        if (height % merge or width % merge or pixels.ndim != 2
                or pixels.shape[0] != height * width
                or counts != [height * width // merge**2]):
            raise ValueError('Qwen image grid, pixel rows, and visual tokens disagree')
        values = {'pixel_values': pixels, 'image_grid_thw': grid}
        fields = {'pixel_values': field.flat_from_sizes('image', grid.prod(-1)),
                  'image_grid_thw': field.batched('image', keep_on_cpu=True)}
    else:
        positions = packet['image_position_ids'].detach().cpu().contiguous()
        if (pixels.ndim != 3 or pixels.shape[0] != 1
                or positions.shape != (*pixels.shape[:2], 2)
                or positions.is_floating_point() or positions.dtype == torch.bool):
            raise ValueError('Gemma image positions and pixel shapes disagree')
        valid = (positions >= 0).all(-1)
        padding = (positions == -1).all(-1)
        if not torch.all(valid | padding):
            raise ValueError('Gemma image patch coordinates contain malformed padding')
        pool = 1 if model_type == 'gemma4_unified' else _positive_int(
            pooling_kernel_size, 'pooling_kernel_size')**2
        valid_counts = valid.sum(-1)
        if torch.any(valid_counts % pool) or counts != (valid_counts // pool).tolist():
            raise ValueError('Gemma image patch positions and visual tokens disagree')
        values = {'pixel_values': pixels, 'pixel_position_ids': positions}
        fields = {key: field.batched('image') for key in values}
    item = MultiModalKwargsItems.from_hf_inputs(values, fields)['image'][0]
    encoded = encode_mm_kwargs_item(item)
    digest = hashlib.sha256()
    for value in (policy_version, model_type, 'image', encoded):
        digest.update(value.encode('utf-8'))
        digest.update(b'\0')
    return {'token_ids': tokens, 'features': {
        'mm_hashes': {'image': [digest.hexdigest()]},
        'mm_placeholders': {'image': [placeholder]},
        'kwargs_data': {'image': [encoded]},
    }}
