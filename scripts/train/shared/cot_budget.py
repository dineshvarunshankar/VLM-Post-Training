"""Resolve medical-only native vLLM reasoning budgets and server requirements."""

FINAL_ANSWER_RESERVE = 1024
TRANSITION_ALLOWANCE = 8
PARSERS = {'qwen3_5':'qwen3','gemma4':'gemma4'}
MODEL_PARSERS = {'qwen3.5-9b':'qwen3','qwen3.8-27b':'qwen3',
                 'gemma4-12b-it':'gemma4','gemma4-31b-it':'gemma4'}


def validate_budget(settings):
    budget = settings['thinking_token_budget']
    if type(budget) is not int or budget < 0:
        raise ValueError('cot.thinking_token_budget must be a non-negative integer')
    if budget + FINAL_ANSWER_RESERVE + TRANSITION_ALLOWANCE > settings['medical_max_new_tokens']:
        raise ValueError('cot.thinking_token_budget must reserve 1024 final-answer tokens plus an 8-token conservative transition allowance within medical_max_new_tokens')


def server_arguments(run):
    if run.get('answer_format','answer_only') != 'cot':
        return []
    if run['model'] not in MODEL_PARSERS:
        raise ValueError('Unsupported model for CoT reasoning budget enforcement')
    return ['--reasoning-parser',MODEL_PARSERS[run['model']]]


def generation_policy(config, *, reasoning_parser):
    validate_budget(config['cot'])
    return {
        'enforcement':'vllm-0.30.0 native SamplingParams.thinking_token_budget',
        'medical_vqa':{'thinking':True,'max_new_tokens':config['cot']['medical_max_new_tokens'],
                       'constraint':'final_answer_json_schema',
                       'thinking_token_budget':config['cot']['thinking_token_budget'],
                       'reasoning_parser':reasoning_parser,'final_answer_reserve':FINAL_ANSWER_RESERVE,
                       'conservative_transition_allowance':TRANSITION_ALLOWANCE},
        'scene_description':{'thinking':False,'constraint':'none','max_new_tokens':config['cot']['caption_max_new_tokens']},
        'camera_view':{'thinking':False,'constraint':'task_json_schema','max_new_tokens':config['evaluation']['max_new_tokens']},
    }
