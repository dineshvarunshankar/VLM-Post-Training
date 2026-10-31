"""Native multimodal prompts with loss restricted to the final JSON response."""

import math

import torch
from scripts.train.shared.video import capped_sampling


class AnswerOnlyCollator:
    def __init__(self, processor, fps, max_length, processor_kwargs=None):
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps:
            raise ValueError("video.fps must be positive")
        self.processor = processor
        self.fps = fps
        self.max_length = max_length
        self.kwargs = dict(processor_kwargs or {})
        reserved = {
            "text",
            "padding",
            "truncation",
            "return_tensors",
            "return_dict",
            "tokenize",
            "add_generation_prompt",
            "enable_thinking",
            "fps",
            "num_frames",
        }
        if reserved & self.kwargs.keys():
            raise ValueError("processor_kwargs contains reserved collator arguments")
        if getattr(self.processor, "video_processor", None) is None:
            raise ValueError("This checkpoint has no native video processor")
        self.processor.video_processor.num_frames = None
        self.processor.tokenizer.padding_side = "right"
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("The model tokenizer must define a padding token")

    @staticmethod
    def messages(example):
        return [
            {"role": "system", "content": example["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": example["video"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
        ]

    def prompt_batch(self, examples, *, padding_side="left"):
        previous = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = padding_side
        try:
            with capped_sampling(
                self.processor,
                self.fps,
                [row.get("required_frame_indices") or [] for row in examples],
                [row.get("video_metadata") for row in examples],
            ):
                return self.processor.apply_chat_template(
                    [self.messages(row) for row in examples],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={
                        "fps": self.fps,
                        "padding": True,
                        "truncation": False,
                        **self.kwargs,
                    },
                )
        finally:
            self.processor.tokenizer.padding_side = previous

    def __call__(self, examples):
        suffixes = []
        for example in examples:
            messages = self.messages(example)
            assistant = {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
            }
            prefix = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = self.processor.apply_chat_template(
                messages + [assistant],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if full.startswith(prefix):
                suffix = full[len(prefix) :]
            else:
                base, found, ending = full.rpartition(example["answer"])
                # Gemma 4's non-thinking generation prefix includes an empty thought channel.
                # Keep that fixed prefix masked; supervise only the JSON and native turn ending.
                if (
                    self.processor.__class__.__name__.startswith("Gemma4")
                    and found
                    and prefix == base + "<|channel>thought\n<channel|>"
                ):
                    suffix = example["answer"] + ending
                else:
                    raise ValueError(
                        "Native assistant boundary is unsupported; refusing to guess the answer mask"
                    )
            if not suffix.startswith(example["answer"]):
                raise ValueError(
                    "Native template inserts content before the final answer; refusing to supervise hidden reasoning"
                )
            target = self.processor.tokenizer(suffix, add_special_tokens=False)[
                "input_ids"
            ]
            if not target:
                raise ValueError("Empty assistant target")
            suffixes.append(target)
        # Decode each video once; the processor retains model-specific frame/timestamp handling.
        batch = self.prompt_batch(examples, padding_side="right")
        input_ids, attention = batch["input_ids"], batch["attention_mask"]
        lengths = attention.sum(dim=1).tolist()
        widths = [int(n) + len(t) for n, t in zip(lengths, suffixes)]
        width = max(widths)
        if width > self.max_length:
            raise ValueError(
                f"Video + prompt + answer needs {width} tokens; maximum is {self.max_length}. "
                "Adjust video FPS/resolution or context limit; inputs are never silently truncated."
            )
        if "position_ids" in batch:
            raise ValueError(
                "Processor returns position_ids; a model-specific position adapter is required"
            )
        fields = [
            key for key in ("token_type_ids", "mm_token_type_ids") if key in batch
        ]
        result_ids = input_ids.new_full(
            (len(examples), width), self.processor.tokenizer.pad_token_id
        )
        result_attention = attention.new_zeros((len(examples), width))
        labels = input_ids.new_full((len(examples), width), -100)
        typed = {key: batch[key].new_zeros((len(examples), width)) for key in fields}
        for i, (length, target) in enumerate(zip(lengths, suffixes)):
            length = int(length)
            selected = attention[i].bool()
            result_ids[i, :length] = input_ids[i, selected]
            result_ids[i, length : widths[i]] = torch.tensor(
                target, dtype=input_ids.dtype, device=input_ids.device
            )
            result_attention[i, : widths[i]] = 1
            labels[i, length : widths[i]] = result_ids[i, length : widths[i]]
            for key in fields:
                typed[key][i, :length] = batch[key][i, selected]
        batch.update(
            input_ids=result_ids,
            attention_mask=result_attention,
            labels=labels,
            **typed,
        )
        return batch
