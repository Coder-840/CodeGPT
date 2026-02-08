from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TheBloke/tiny-llama-7B-GPTQ"  # Tiny 8-bit quantized version

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = None

def load_model():
    global model
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_8bit=True
        )
    return model

def generate(prompt, max_length=256):
    model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
