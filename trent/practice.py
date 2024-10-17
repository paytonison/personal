from transformers import pipeline
import io
import os
import torch

f1 = open("practice.txt", "w")
f2 = open("practice.txt", "r")

def generate(f1, f2):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
            {"role": "system", "content": f"You are an autonomous artificial intelligence agent tasked with writing every thought in your head. {f2.read()}"},
            {"role": "user", "content": f2.read()},
        ]
    
    outputs = pipe(messages, max_new_tokens=256)
    return outputs

while True:
    huh = generate(f1, f2)
    f1.write(huh[0]["generated_text"][2]["content"])
