import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        
class DQLearningAgent:
    def __init__(self, state_size, action_size, hidden_size, learning_rate=0.001, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.model = CustomRNN(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        
    def choose_action(self, state, hidden):
        if torch.rand(1).item < self.epsilon:
            return torch.randint(0, self.action_size, (1,)).item(), hidden
        else:
            q_values, hidden = self.model(state, hidden)
            return torch.argmax(q_values).item(), hidden
        
    
    def train(self, state, action, reward, next_state, done, hidden):
        q_values, hidden = self.model(state, hidden)
        q_value = q_values[0, action]
        
        with torch.no_grad():
            next_q_values, _ = self.model(next_state, hidden)
            max_next_q_value = torch.max(next_q_values)
            target = reward + (1 - done) * self.gamma * max_next_q_value
        
        loss = F.mse_loss(q_value, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text_data, vocab):
        self.text_data = [torch.tensor([vocab[word] for word in sentence]) for sentence in text_data]
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        return self.text_data[idx]


def train_agent(agent, text_loader, num_epochs):
    for epoch in range(num_epochs):
        hidden = agent.model.init_hidden(batch_size=1)
        total_loss = 0
        for state, next_state, reward, done in text_loader:
            action, hidden = agent.choose_action(state, hidden)
            agent.train(state, action, reward, next_state, done, hidden)
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(text_loader)}')