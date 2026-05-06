import json

class DataProcessor:
    def __init__(self, input_jsonl_file):
        self.input_file = input_jsonl_file

    def process_and_save(self, output_file="outputs/dataset/sft_model_name.jsonl"):

        with open(self.input_file, 'r') as f_in, \
             open(output_file, 'w') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                    
                data = json.loads(line)

                # Format the reasoning block 
                cot = data.get('cot', '')
                answer = data.get('answer', '').capitalize()
                reasoning_answer = f"<|channel>thought\n{cot}\n<channel|>\n{answer}."

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
                
        print(f"Data conversion complete. Saved to: {output_file}")
        return output_file

if __name__ == "__main__":
    processor = DataProcessor("exports/outputs/sft.jsonl")
    processor.process_and_save()