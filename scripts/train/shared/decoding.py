"""Category JSON constraints and model-author sampling settings for evaluation."""

import json

import torch
import xgrammar as xgr
from scripts.train.shared.data import label_keys
from transformers import LogitsProcessor, LogitsProcessorList
from xgrammar.contrib.hf import LogitsProcessor as GrammarLogitsProcessor


def answer_schema(definition):
    keys = label_keys(definition)
    return {
        "type": "object",
        "properties": {key: {"type": "integer", "enum": [0, 1]} for key in keys},
        "required": keys,
        "additionalProperties": False,
    }


class PresencePenalty(LogitsProcessor):
    """Apply the additive presence penalty to generated tokens only."""

    def __init__(self, penalty, prompt_length):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length :]
        if generated.numel():
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, generated, True)
            scores = scores - self.penalty * seen
        return scores


class ConstrainedAnswers:
    def __init__(self, tokenizer, model_config, generation_config, catalog):
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=model_config.get_text_config().vocab_size,
            stop_token_ids=generation_config.eos_token_id,
        )
        self.compiler = xgr.GrammarCompiler(info)
        self.catalog = catalog
        self.compiled = {}

    def processors(self, categories, prompt_length, presence_penalty=0.0):
        grammars = []
        for category in categories:
            if category not in self.compiled:
                schema = answer_schema(self.catalog["categories"][category])
                self.compiled[category] = self.compiler.compile_json_schema(
                    json.dumps(schema), any_whitespace=False, separators=(",", ":")
                )
            grammars.append(self.compiled[category])
        processors = LogitsProcessorList()
        if presence_penalty:
            processors.append(PresencePenalty(presence_penalty, prompt_length))
        processors.append(GrammarLogitsProcessor(grammars))
        return processors
