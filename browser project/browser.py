from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from transformers import pipeline
import json
import time
import torch
import re

# Initialize the WebDriver (replace with your path to ChromeDriver)
# driver = webdriver.Safari()

pipe = pipeline(
"text-generation",
"meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    max_new_tokens=256
)

def action(pipe):
    messages = [
    {"role": "system", "content": '''
            You are an autonomous web agent. As you formulate your answer, create a chain of thought.\n 
            Respond in a valid JSON format like: {action: action_type, query: some_query_or_url}\n
            In your reponse, also include the website you are searching on in a similarly valid JSON format like: {website: website_url}'''},
    {"role": "user", "content": "Search for something you like."}
    ]
    action = pipe(messages)
    return action[0]["generated_text"][-1]["content"]


bot_action = action(pipe)
print(bot_action)