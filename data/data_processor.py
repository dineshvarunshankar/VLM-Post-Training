import json

class DataProcessor:
    """
    Processes raw JSONL data into model-specific chat format.
    
    Supports multiple model reasoning formats:
        - gemma4: Uses <|channel>thought / <channel|> tokens (Gemma-4 thinking format)
        - cosmos: Uses <think> / </think> and <answer> / </answer> tags (Cosmos-Reason2 / Qwen format)
    """

    reasoning_formats = {
        "gemma4": "<|channel>thought\n{cot}\n<channel|>\n{answer}.",
        "cosmos": "<think>\n{cot}\n</think>\n\n<answer>\n{answer}\n</answer>",
    }

    def __init__(self, input_jsonl_file, model_type):
       
        self.input_file = input_jsonl_file
        self.model_type = model_type

    def _format_reasoning(self, cot, answer):
        """Format the chain-of-thought reasoning block with model-specific tokens."""
        return self.reasoning_formats[self.model_type].format(
            cot=cot, answer=answer.capitalize()
        )

    def process_and_save(self, output_file=None):
        """
        Reads raw JSONL, wraps each sample into HuggingFace messages format with
        model-specific reasoning tokens, and writes to output JSONL.
        """
        if output_file is None:
            output_file = f"outputs/dataset/sft_{self.model_type}.jsonl"

        with open(self.input_file, 'r') as f_in, \
             open(output_file, 'w') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                    
                data = json.loads(line)

                # Format the reasoning block with model-specific tokens
                cot = data.get('cot', '')
                answer = data.get('answer', '')
                reasoning_answer = self._format_reasoning(cot, answer)

                # Create the prompt instructions based on the model type
                if self.model_type == "cosmos":
                    prompt_instruction = (
                        f"{data['question']} Answer the question using the following format:\n\n"
                        "<think>\nFirst, output the bounding box coordinates of the subject in the green box. Then, provide your step-by-step reasoning.\n</think>\n\n"
                        "<answer>\nYour final answer.\n</answer>"
                    )
                else: # gemma4
                    prompt_instruction = (
                        f"{data['question']} First, identify and output the bounding box coordinates of the subject in the green box. Then, provide your step-by-step reasoning before answering."
                    )

                # Structure into Hugging Face nested dictionary format
                formatted_data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": data['image']
                                },
                                {
                                    "type": "text",
                                    "text": prompt_instruction
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": reasoning_answer
                                }
                            ]
                        }
                    ]
                }
                
                # Write the formatted data as a JSONL string
                f_out.write(json.dumps(formatted_data) + "\n")
                
        print(f"[{self.model_type}] Data conversion complete. Saved to: {output_file}")
        return output_file

    def process_for_grpo(self, output_file=None):
        """
        Reads raw JSONL, wraps each sample into HuggingFace messages format with
        model-specific reasoning tokens, and preserves ground_truth.
        """
        if output_file is None:
            output_file = f"outputs/dataset/grpo_{self.model_type}.jsonl"

        with open(self.input_file, 'r') as f_in, \
             open(output_file, 'w') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                    
                data = json.loads(line)

                # Create the prompt instructions based on the model type
                if self.model_type == "cosmos":
                    prompt_instruction = (
                        f"{data['question']} Answer the question using the following exact format:\n\n"
                        "<think>\n[Provide the subject bounding box as [x1, y1, x2, y2], then reason to a yes/no conclusion]\n</think>\n\n"
                        "<answer>\n[Your final answer]\n</answer>"
                        "\n\nThe final answer must be exactly yes or no."
                    )
                else: # gemma4
                    prompt_instruction = (
                        f"{data['question']} First, provide the bounding box coordinates of the subject in the green box. Then, provide your reasoning on how to solve the question before giving your final answer."
                    )

                # Structure into Hugging Face nested dictionary format with ground_truth
                formatted_data = {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": data["image"],
                                },
                                {
                                    "type": "text",
                                    "text": prompt_instruction
                                }
                            ]
                        }
                    ],
                    "ground_truth": {
                        "answer": data.get('answer', ''),
                        "cot": data.get('cot', ''),
                        "subject_bbox": data.get('subject_bbox', [])
                    }
                }
                
                # Write the formatted data as a JSONL string
                f_out.write(json.dumps(formatted_data) + "\n")
                
        print(f"[{self.model_type}] GRPO data conversion complete. Saved to: {output_file}")
        return output_file

    def process_and_save_no_cot(self, output_file=None):
        if output_file is None:
            output_file = f"outputs/dataset/sft_{self.model_type}_no_cot.jsonl"

        with open(self.input_file, 'r') as f_in, \
             open(output_file, 'w') as f_out:

            for line in f_in:
                if not line.strip():
                    continue

                data = json.loads(line)
                answer = str(data.get('answer', '')).strip()
                answer_line = answer.capitalize() if answer else ""

                if self.model_type == "cosmos":
                    prompt_instruction = (
                        f"{data['question']} Do not include reasoning or chain-of-thought. "
                        "Answer using exactly this format:\n\n"
                        "<answer>\n[yes or no]\n</answer>\n\n"
                        "The final answer must be exactly yes or no."
                    )
                    assistant_text = f"<answer>\n{answer_line}\n</answer>"
                else:
                    prompt_instruction = (
                        f"{data['question']} Answer directly with no explanation or reasoning."
                    )
                    assistant_text = f"{answer_line}." if answer_line else ""

                formatted_data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": data['image']
                                },
                                {
                                    "type": "text",
                                    "text": prompt_instruction
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": assistant_text
                                }
                            ]
                        }
                    ]
                }

                f_out.write(json.dumps(formatted_data) + "\n")

        print(f"[{self.model_type}] SFT no-CoT data conversion complete. Saved to: {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage for both models:
    # processor = DataProcessor("outputs/dataset/sft.jsonl", model_type="gemma4")
    # processor.process_and_save()
    
    # processor = DataProcessor("outputs/dataset/sft.jsonl", model_type="cosmos")
    # processor.process_and_save()
    
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
    processor = DataProcessor("outputs/dataset/sft.jsonl", model_type=model_type)
    processor.process_and_save()
