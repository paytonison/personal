import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Step 1: Load and Preprocess Data
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(1 if label == 'pos' else 0)
        processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.float32).to(device), pad_sequence(text_list, batch_first=True).to(device)

train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
train_dataloader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)

# Step 2: Define the Model
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return torch.sigmoid(self.fc(hidden[-1]))

# Initialize the model
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 1
model = RNNTextClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

# Step 3: Training Setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for labels, texts in train_dataloader:
        optimizer.zero_grad()
        predictions = model(texts).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Save the model after training
torch.save(model.state_dict(), "rnn_text_classifier.pth")
print("Model saved as rnn_text_classifier.pth")

