from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import deque
import torch
import random

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def get_state(text):
    return tokenizer.encode(text, return_tensors="pt")


def take_action(state, action):
    input_ids = torch.cat((state, torch.tensor([[action]])), dim=1)
    return input_ids


def get_reward(output_text):
    if "coherent" in output_text:
        return 10
    return 0


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    
replay_buffer = ReplayBuffer(capacity=1000)

learning_rate = 1e-4
batch_size = 16
gamma = 0.99
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_q_network():
    batch = replay_buffer.sample(batch_size)
    
    for state, action, reward, next_state in batch:
        input_ids = take_action(state, action)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        next_q_value = max([model(get_state(next_state), labels=get_state(next_state)).loss.item() for _ in range(5)])
        target = reward + gamma * next_q_value
        
        q_value = -loss
        loss = (q_value - target) ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
for episode in range(100):
    state = get_state("The quick brown fox")
    done = False
    
    while not done:
        action = random.randint(0, model.config.vocab_size - 1)
        
        next_state = take_action(state, action)
        output_text = tokenizer.decode(next_state[0])
        
        reward = get_reward(output_text)
        
        replay_buffer.add((state, action, reward, output_text))
        
        if len(replay_buffer.buffer) > batch_size:
            train_q_network()
            
        state = next_state
        
        if len(state[0]) > 20:
            done = True