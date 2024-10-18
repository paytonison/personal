# Make sure to reinitialize the model with the same structure
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

# Load the saved model
model = RNNTextClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("/Users/paytonison/personal git/custom nn/rnn_text_classifier.pth"))
model.eval()  # Switch to evaluation mode

# Evaluation function
def evaluate_model(dataloader):
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for labels, texts in dataloader:
            predictions = model(texts).squeeze()
            predicted_labels = (predictions >= 0.5).float()  # Convert to 0 or 1
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

# Running evaluation on the test set
test_accuracy = evaluate_model(test_dataloader)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
