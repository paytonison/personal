import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class TextEnvironment:
    def __init__(self):
        self.base_prompts = [
            "Hello, how can I help you with {}?",
            "Do you have any questions about {}?",
            "What is your opinion on {}?"
        ]
        self.topics = ["technology", "Python", "reinforcement learning"]
        self.action_space = [
            "Python is a programming language.",
            "I don't understand your question.",
            "Reinforcement learning is a type of machine learning."
        ]
        self.state = None

    def reset(self):
        base_prompt = random.choice(self.base_prompts)
        topic = random.choice(self.topics)
        self.state = base_prompt.format(topic)
        return self.state

    def step(self, action):
        # Extract the topic from the current state
        if "technology" in self.state:
            topic = "technology"
        elif "Python" in self.state:
            topic = "Python"
        elif "reinforcement learning" in self.state:
            topic = "reinforcement learning"
        else:
            topic = None

        # Determine if the action is contextually appropriate for the current state
        if self.state.startswith("Do you have any questions about") and action == "Reinforcement learning is a type of machine learning." and topic == "reinforcement learning":
            reward = 1  # Correct and contextually appropriate
        elif self.state.startswith("What is your opinion on") and action == "I don't understand your question." and topic == "Python":
            reward = 0.5
        elif self.state.startswith("Hello, how can I help you with") and action != "I don't understand your question.":
            reward = 0.75
        else:
            reward = 0  # Incorrect or contextually inappropriate

        next_state = self.reset()
        return next_state, reward

# 2. Define the LLaMA 3.2 Model Integration
class LLaMAQNetwork:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        # Set do_sample=True to introduce variability and add temperature for control
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            max_length=50,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Print the generated output for debugging
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract hidden states from the model outputs
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states
        embeddings = hidden_states[-1][:, 0, :].to(device)  # CLS token
        # Normalize embeddings to unit vectors
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(self, state_text):
        # Generate embeddings (no Q-values, but we adapt the function interface)
        return self.embed_text(state_text)

# 3. Define the Epsilon-Greedy Action Selection Function
def select_action(state_text, q_network, epsilon, num_actions, env):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        with torch.no_grad():
            state_embedding = q_network.forward(state_text)
            action_embeddings = [
                q_network.embed_text(action_text) for action_text in env.action_space
            ]

            # Normalize embeddings
            state_embedding = torch.nn.functional.normalize(state_embedding, p=2, dim=1)
            action_embeddings = [
                torch.nn.functional.normalize(embedding, p=2, dim=1)
                for embedding in action_embeddings
            ]

            # Measure similarity between state and each action
            similarities = [
                cosine_similarity(state_embedding, action_embedding, dim=1).item()
                for action_embedding in action_embeddings
            ]
            
            # Check if there is a meaningful difference in similarity
            best_action_index = torch.tensor(similarities).argmax().item()
            max_similarity = similarities[best_action_index]

            # Add slight randomness if all similarities are very close (e.g., near 1.0)
            if max_similarity >= 0.99 and all(abs(sim - max_similarity) < 0.01 for sim in similarities):
                return random.randint(0, num_actions - 1)

            # Ensure the action is contextually relevant
            if similarities[best_action_index] < 0.5:
                return random.randint(0, num_actions - 1)

            return best_action_index

# 4. Main Training Loop with Action Printing
def train_dqn():
    # Hyperparameters
    learning_rate = 0.0001
    gamma = 0.99
    epsilon = 0.4
    num_actions = 3
    num_episodes = 1000

    # Initialize LLaMA model, optimizer, and environment
    q_network = LLaMAQNetwork()
    optimizer = optim.SGD(q_network.model.parameters(), momentum=0.9, lr=learning_rate)
    env = TextEnvironment()

    # Training Loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(10):
            action_index = select_action(state, q_network, epsilon, num_actions, env)
            action = env.action_space[action_index]  # Convert action index to actual action text
            next_state, reward = env.step(action)
            total_reward += reward

            # Print the state, action taken, and reward received
            print(f"Episode {episode + 1}, Step {step + 1}")
            print(f"State: {state}")
            print(f"Generated Action: {action}")
            print(f"Reward: {reward}\n")

            # Update Q-network
            with torch.no_grad():
                next_state_embedding = q_network.forward(next_state)
                action_embeddings = [
                    q_network.embed_text(action_text) for action_text in env.action_space
                ]
                next_q_values = [
                    cosine_similarity(next_state_embedding, action_embedding, dim=1).item()
                    for action_embedding in action_embeddings
                ]

                max_next_q_value = max(next_q_values)
                target = reward + gamma * max_next_q_value

            # Calculate loss and backpropagate
            state_embedding = q_network.forward(state)
            q_values = [
                cosine_similarity(state_embedding, action_embedding, dim=1)
                for action_embedding in action_embeddings
            ]
            q_value = q_values[action_index]

            # Ensure target and q_value are PyTorch tensors for gradient calculation
            target_tensor = torch.tensor(target, requires_grad=False).to(device)
            q_value_tensor = torch.tensor(q_value, requires_grad=True).to(device)

            # Calculate loss
            loss = (target_tensor - q_value_tensor) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move to the next state
            state = next_state

        print(f"Episode {episode + 1} Complete - Total Reward: {total_reward}")
        print("=" * 50)  # Separate episodes for clarity

# Start the training process
if __name__ == "__main__":
    train_dqn()

