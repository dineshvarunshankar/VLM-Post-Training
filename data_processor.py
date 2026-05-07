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
                                    "text": data['question']
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