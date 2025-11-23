import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ArctosEngine:
    def __init__(self):
        print("⛄ Loading ARCTOS Mixtral engine…")
        model_path = "./models/mixtral-8x7b"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
