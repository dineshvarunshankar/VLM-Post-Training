"""Pinned Qwen/Gemma thinking boundaries and strict final-answer extraction."""

from dataclasses import dataclass
import json

FORMATS = {
    'qwen3_5': {'opening':'<think>\n', 'boundary':'\n</think>\n\n', 'ending':'<|im_end|>'},
    'gemma4': {'opening':'<|channel>thought\n', 'boundary':'\n<channel|>', 'ending':'<turn|>'},
}
TASK_IDS = {'medical_vqa':0, 'scene_description':1, 'camera_view':2}


def model_family(config):
    kind = getattr(config, 'model_type', None)
    if kind == 'qwen3_5':
        return kind
    if kind in ('gemma4', 'gemma4_unified'):
        return 'gemma4'
    raise ValueError(f'Unsupported pinned CoT model family: {kind}')


@dataclass(frozen=True)
class ExtractedAnswer:
    answer: str | None
    diagnostics: dict


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Repeated JSON key')
        result[key] = value
    return result


def extract_cot_answer(text, *, model_family, finish_reason=None):
    """Parse only the final native channel; never scan reasoning for answer JSON.

    Qwen's opening thought token is already in the prompt, so its completion
    starts with reasoning. Gemma emits its thought-channel opening itself.
    A server-confirmed stop may omit its stop token; a bare string must include
    the native turn ending. A length stop is always reported as truncated.
    """
    if model_family not in FORMATS:
        raise ValueError(f'Unsupported CoT model family: {model_family}')
    fmt = FORMATS[model_family]
    diagnostics = dict(malformed=True, empty_reasoning=False, truncated=finish_reason=='length',
                       finish_reason=finish_reason, characters=len(text), reason=None)
    def invalid(reason):
        diagnostics['reason'] = reason
        return ExtractedAnswer(None, diagnostics)
    if diagnostics['truncated']:
        return invalid('generation budget exhausted')
    body = text.rstrip()
    if body.endswith(fmt['ending']):
        body = body[:-len(fmt['ending'])]
    elif finish_reason != 'stop':
        diagnostics['truncated'] = True
        return invalid('missing native turn ending')
    boundary = '</think>' if model_family=='qwen3_5' else '<channel|>'
    if body.count(boundary) != 1:
        return invalid('missing or repeated reasoning-to-answer boundary')
    reasoning, answer = body.split(boundary)
    answer = answer.strip()
    opening = '<think>' if model_family=='qwen3_5' else fmt['opening']
    if reasoning.startswith(opening):
        reasoning = reasoning[len(opening):]
    elif model_family == 'gemma4':
        return invalid('missing native thought-channel opening')
    diagnostics['empty_reasoning'] = not reasoning.strip()
    diagnostics['reasoning_characters'] = len(reasoning)
    if diagnostics['empty_reasoning']:
        return invalid('empty reasoning')
    for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>', '<|turn>', '<turn|>', '<|channel>', '<channel|>'):
        if token in reasoning or token in answer:
            return invalid('unexpected native control token')
    try:
        value = json.loads(answer, object_pairs_hook=unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        if not isinstance(value, dict) or not value:
            return invalid('final answer must be one nonempty JSON object')
    except (ValueError, TypeError):
        return invalid('final channel must contain exactly one valid JSON object')
    diagnostics['malformed'] = False
    return ExtractedAnswer(json.dumps(value, separators=(',', ':'), ensure_ascii=False), diagnostics)


def assistant_suffix(processor, example, *, family):
    from scripts.train.shared.collator import AnswerOnlyCollator
    fmt = FORMATS[family]
    medical = example['task'] == 'medical_vqa'
    if example.get('teacher_available') is False or (medical and not example.get('reasoning')):
        raise ValueError('Teacher target unavailable; use held-out human GT only for generation scoring')
    if not isinstance(example['answer'], str) or not example['answer'].strip():
        raise ValueError('Empty assistant target')
    messages = AnswerOnlyCollator.messages(example)
    assistant = {'role':'assistant', 'content':[{'type':'text','text':example['answer']}]}
    if medical:
        assistant['reasoning_content'] = example['reasoning']
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=medical)
    full = processor.apply_chat_template(messages+[assistant], tokenize=False, add_generation_prompt=False, enable_thinking=medical)
    if full.startswith(prefix):
        suffix = full[len(prefix):]
    elif not medical and family == 'gemma4':
        base, found, ending = full.rpartition(example['answer'])
        if not found or prefix != base + '<|channel>thought\n<channel|>':
            raise ValueError('Unsupported native assistant boundary')
        suffix = example['answer'] + ending
    else:
        raise ValueError('Unsupported native assistant boundary; template differs from pinned evidence')
    if medical:
        expected = ((example['reasoning'] if family=='qwen3_5' else fmt['opening']+example['reasoning'])
                    + fmt['boundary'] + example['answer'] + fmt['ending'] + '\n')
        if suffix != expected:
            raise ValueError('Pinned template did not preserve the complete reasoning, final answer, and native turn ending')
        parsed = extract_cot_answer(suffix, model_family=family)
        if parsed.answer != json.dumps(json.loads(example['answer']),separators=(',', ':'),ensure_ascii=False):
            raise ValueError('Teacher rationale contains an invalid native boundary')
    elif suffix != example['answer'] + fmt['ending'] + '\n':
        raise ValueError('Direct target has unexpected thinking content or native ending')
    return prefix, suffix
