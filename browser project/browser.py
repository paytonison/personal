from selenium import webdriver
from transformers import pipeline
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto")

# Initialize the browser (using Chrome in this case)
driver = webdriver.Safari()

# A simple command interpreter for browser actions
class BrowserAI:
    def __init__(self, driver):
        self.driver = driver

    def perform_action(self, query):
        messages = [
                {"role": "system", "content": 
                '''You are an autonomous AI assistant with agentic abilities designed to search the internet.\n
                Your task is to complete the requested task by searching the internet based on the user's input.\n
                You return results in a structured format, like this:
                {"type": "navigate", "url": "https://google.com"}\n
                Being token-efficient: avoid returning excessively long outputs.\n'''},
                {"role": "user", "content": query}
            ]
        action = pipe(
        messages,
        max_new_tokens=20,
        )
        print(action)
        if action[0] == 'navigate':
            self.navigate(action['url'])
        elif action[0] == 'search':
            self.search(action['query'])
        elif action[0] == 'click':
            self.click(action['selector'])
        elif action[0] == 'type':
            self.type_text(action['selector'], action['text'])
        else:
            print("Unknown action:", action)

    def navigate(self, url):
        self.driver.get(url)

    def search(self, query):
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
    
    def click(self, selector):
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.click()

    def type_text(self, selector, text):
        element = self.driver.find_element(By.CSS_SELECTOR, selector)
        element.clear()
        element.send_keys(text)

browser = BrowserAI(driver)

browser.perform_action("Search for the best restaurants in New York City")
