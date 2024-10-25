import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("mps")
model.to(device)

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through three linear layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def prepare_input(text): # tokenizing
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return inputs.input_ids.to(device), inputs.attention_mask.to(device)

# Load a sentence transformer for calculating semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_reward(generated_text, reference_texts):
    # Initialize individual reward components
    semantic_coherence_reward = 0.0
    stylistic_matching_reward = 0.0
    syntactic_variety_reward = 0.0
    philosophical_tone_reward = 0.0
    diversity_penalty = 0.0
    
    # 1. Semantic Coherence
    generated_embedding = semantic_model.encode(generated_text, convert_to_tensor=True)
    reference_embedding = semantic_model.encode(reference_texts, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(generated_embedding, reference_embedding).item()
    
    if semantic_similarity > 0.7:  # Threshold for semantic coherence
        semantic_coherence_reward = 1.0

    # 2. Stylistic Matching - Check for specific Nietzschean phrases or patterns
    if any(phrase in generated_text.lower() for phrase in ["will to power", "eternal return", "overman", "beyond good and evil", "god is dead", "ubermensch"]):
        stylistic_matching_reward = 0.5

    # 3. Syntactic Variety - Encourage more complex sentence structures
    if len(generated_text.split()) > 15 and (';' in generated_text or '—' in generated_text):
        syntactic_variety_reward = 1.0

    # 4. Philosophical Tone - Reward for philosophical language use
    philosophical_keywords = ["truth", "power", "moral", "chaos", "being", "essence", "doubt", "freedom", "apollonian", "dionysian", "eternity", "nihilism", "eternal", "will", "abyss", "void"]
    for word in philosophical_keywords:
        if word in generated_text.lower():
            philosophical_tone_reward += 0.2

    # 5. Refined Diversity Penalty - Proportional and less aggressive
    token_list = generated_text.split()
    unique_token_ratio = len(set(token_list)) / len(token_list)

    # Adjust threshold and scale penalty
    if unique_token_ratio < 0.4:  # Make it less strict
        diversity_penalty = -0.2 * (0.8 - unique_token_ratio)  # Proportional penalty

    # Ensure that diversity penalty does not dominate the reward
    diversity_penalty = max(diversity_penalty, -0.5)  # Cap the maximum penalty

    # Total reward calculation
    total_reward = (semantic_coherence_reward +
                    stylistic_matching_reward +
                    syntactic_variety_reward +
                    philosophical_tone_reward +
                    diversity_penalty)

    # Print detailed breakdown
    print(f"Reward Breakdown:")
    print(f"  Semantic Coherence: {semantic_coherence_reward}")
    print(f"  Stylistic Matching: {stylistic_matching_reward}")
    print(f"  Syntactic Variety: {syntactic_variety_reward}")
    print(f"  Philosophical Tone: {philosophical_tone_reward}")
    print(f"  Diversity Penalty: {diversity_penalty}")
    print(f"  Total Reward: {total_reward}")

    return total_reward

# Define hyperparameters
input_dim = 768  # GPT-2 embedding size
hidden_dim = 512
output_dim = tokenizer.vocab_size  # Number of possible actions (vocabulary size)
learning_rate = 0.01

# Initialize simpler Q-Network
q_network = QNetwork(input_dim, hidden_dim, output_dim).to(device)

# Use SGD as before, but with the simpler Q-network
optimizer = optim.SGD(q_network.parameters(), lr=learning_rate)

loss_fn = nn.HuberLoss(delta=1.0)

epochs = 33
for epoch in range(epochs):
    model.train()
    reference_text = 'Supposing that Truth is a woman—what then? Is there not ground for suspecting that all philosophers, in so far as they have been dogmatists, have failed to understand women—that the terrible seriousness and clumsy importunity with which they have usually paid their addresses to Truth have been unskilled and unseemly methods for winning a woman? Certainly, she has never allowed herself to be won; and at present, every kind of dogma stands with sad and discouraged mien—IF, indeed, it stands at all! For there are scoffers who maintain that it has fallen, that all dogma lies on the ground—nay more, that it is at its last gasp. But to speak seriously, there are good grounds for hoping that all dogmatizing in philosophy may have been only a noble puerilism and tyronism. Perhaps some play upon words, a deception on the part of grammar, or an audacious generalization of very restricted, very personal, very human—all-too-human facts.' 
    # Generate initial text from the transformer model
    # Prepare initial input
    input_ids, attention_mask = prepare_input("In the vast silence of the universe, where light and shadow dance eternally, there lies a truth that few dare to speak. The will to power is not merely a force—it is the essence of life, surging through the veins of existence, driving every creature, every thought, every whisper of defiance. But what, then, is the meaning of such power, when the abyss gazes back?")
    outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,         # Allow for longer generation
    num_return_sequences=1, # Generate one output at a time
    pad_token_id=model.config.pad_token_id,
    do_sample=True,         # Enable sampling for variety
    top_k=40,               # Explore top-40 tokens
    top_p=0.85,             # Use nucleus sampling for diversity
    temperature=0.8,        # Slightly increase temperature for stylistic variation
    no_repeat_ngram_size=3)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated Text: {generated_text}")

    reward = calculate_reward(generated_text, reference_text)

    # Convert generated text to embeddings
    inputs_embeds = model.transformer.wte(input_ids)  # Embedding layer from GPT-2

    # Pass embeddings through the simpler Q-network (reshape if needed)
    q_values = q_network(inputs_embeds.view(-1, input_dim))

    # Compute loss and backpropagate
    target = torch.tensor([reward]).to(device)
    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Reward: {reward}")

print(f"Epoch average: {reward/epochs}")
