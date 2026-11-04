"""Image-only processor budgets; video options never govern still images."""


def image_options(processor, kwargs):
    image = getattr(processor, 'image_processor', None)
    if image is None:
        raise ValueError('This checkpoint has no native image processor')
    name = type(image).__name__
    if name.startswith(('Qwen2VL', 'Qwen3VL')):
        if set(kwargs) - {'size'}:
            raise ValueError('Qwen image_processor_kwargs supports only size pixel bounds')
        if 'size' in kwargs:
            size = kwargs['size']
            if (not isinstance(size, dict) or set(size) != {'shortest_edge', 'longest_edge'}
                    or any(type(value) is not int or value <= 0 for value in size.values())
                    or size['shortest_edge'] > size['longest_edge']):
                raise ValueError('Image size requires positive ordered shortest_edge/longest_edge pixel bounds')
    elif name.startswith('Gemma4'):
        if set(kwargs) - {'max_soft_tokens'}:
            raise ValueError('Gemma image_processor_kwargs supports only max_soft_tokens')
        if 'max_soft_tokens' in kwargs:
            tokens = kwargs['max_soft_tokens']
            if type(tokens) is not int or tokens not in (70, 140, 280, 560, 1120):
                raise ValueError('Gemma image max_soft_tokens must be 70, 140, 280, 560, or 1120')
    else:
        raise ValueError(f'Unsupported native image processor: {name}')
    return {'images_kwargs': dict(kwargs)} if kwargs else {}
