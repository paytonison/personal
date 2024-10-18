import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("mps")

# Step 1: Load and Preprocess the SMS Spam Data
class SMSDataset(Dataset):
    def __init__(self, filepath):
        # Load the data using the correct separator and only take the first two columns
        self.data = pd.read_csv(filepath, sep=',', usecols=[0, 1], names=["label", "message"], encoding='ISO-8859-1')
        self.data.dropna(subset=["message"], inplace=True)  # Remove rows where 'message' is NaN
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.build_vocab()

    def build_vocab(self):
        # Tokenize the dataset to build the vocabulary
        tokens = [self.tokenizer(str(text)) for text in self.data['message']]
        vocab = build_vocab_from_iterator(tokens, specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = 1 if self.data.iloc[index, 0] == 'spam' else 0
        text = torch.tensor(self.vocab(self.tokenizer(str(self.data.iloc[index, 1]))), dtype=torch.int64)
        return torch.tensor(label, dtype=torch.float32), text

# Step 2: Define the Model (Same as Before)
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

def collate_batch(batch):
    label_list, text_list = [], []
    for (label, text) in batch:
        label_list.append(label)
        text_list.append(text)
    return torch.tensor(label_list, dtype=torch.float32).to(device), pad_sequence(text_list, batch_first=True).to(device)

# Testing Loop
def evaluate_model(dataloader):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for labels, texts in dataloader:
            predictions = model(texts).squeeze()
            predicted_labels = (predictions >= 0.5).float()  # Convert to binary labels: 0 or 1
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

# Initialize Dataset and DataLoaders
sms_dataset = SMSDataset(filepath="/Users/paytonison/personal git/custom nn/spam.csv")
print(f"Total samples after cleaning: {len(sms_dataset)}")

train_size = int(0.8 * len(sms_dataset))
test_size = len(sms_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(sms_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

# Initialize and Train the Model Freshly
vocab_size = len(sms_dataset.vocab)
embed_dim = 128
hidden_dim = 256
output_dim = 1
model = RNNTextClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for labels, texts in train_dataloader:  # Fixed indentation here
        optimizer.zero_grad()
        predictions = model(texts).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Running the evaluation on the test set
test_accuracy = evaluate_model(test_dataloader)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
