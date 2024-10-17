import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-2 Model and Tokenizer
model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

# Set the pad token ID to eos_token_id for consistency
tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

class GPT2EnvWrapper(gym.Env):
    def __init__(self, model, tokenizer):
        super(GPT2EnvWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

        # Define the observation and action space (adjusted vocab size)
        self.action_space = Discrete(50257)  # Size of GPT-2 vocab
        self.observation_space = Box(low=0, high=50256, shape=(1,), dtype=np.int32)

        self.prompt = "Start: "  # Initial prompt to get the GPT-2 generating text
        self.state = self.tokenizer.encode(self.prompt, return_tensors='pt')
        self.max_steps = 10
        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        # Add the action (token) to the current state
        inputs = torch.cat((self.state, torch.tensor([[action]])), dim=1)
        inputs = inputs.to(device)
        # Create attention_mask to prevent unexpected behavior
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        # Generate next token
        outputs = self.model.generate(
            inputs, 
            attention_mask=attention_mask, 
            max_length=inputs.size(1) + 1, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        next_token = outputs[:, -1]
        decoded_text = self.tokenizer.decode(next_token)

        # Check if action (generated text) aligns with an environment-specific goal
        reward = self.calculate_reward(decoded_text)
        
        # Update state with the newly generated token
        self.state = inputs

        # If we've reached the max number of steps, end the episode
        done = self.current_step >= self.max_steps

        # Return next state (token), reward, done, and an empty info dict
        return next_token.item(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.state = self.tokenizer.encode(self.prompt, return_tensors='pt')
        return self.state.squeeze()[-1].item()  # Return the last token as the initial state

    def calculate_reward(self, text):
        # Simple reward logic: reward based on some pattern in the text
        if "success" in text:
            return 1
        else:
            return -1

# Initialize environment with GPT-2
env = GPT2EnvWrapper(model, tokenizer)
state = env.reset()

done = False
total_reward = 0

while not done:
    # GPT-2 generates the next action (token) based on the current state
    action_token = model.generate(
        torch.tensor([[state]]), 
        max_length=2, 
        pad_token_id=tokenizer.eos_token_id
    )[:, -1]
    
    # Step through the environment
    next_state, reward, done, _ = env.step(action_token.item())
    
    total_reward += reward
    state = next_state

print(f'Total Reward: {total_reward}')

