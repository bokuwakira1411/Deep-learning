import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_lstm_layers, n_hidden_layers, n_hidden_nodes, activation, epoch,
                 dropout, learning_rate=0.0001, patience=10, num_classes=10):
        super(BiLSTM, self).__init__()
        self.epoch = epoch
        self.patience = patience
        self.learning_rate = learning_rate
        self.activation = activation
        self.n_lstm_layers = n_lstm_layers
        self.lstm = nn.ModuleList()
        self.batch_norms_lstm = nn.ModuleList()
        self.batch_norms_fc = nn.ModuleList()

        self.lstm.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True,bidirectional=True))
        #self.batch_norms_lstm.append(nn.BatchNorm1d(hidden_sizes[0]*2))
        self.dropouts = nn.ModuleList([nn.Dropout(dropout[i]) for i in range(n_lstm_layers + n_hidden_layers)])

        for i in range(1, n_lstm_layers):
            self.lstm.append(nn.LSTM(hidden_sizes[i - 1]*2, hidden_sizes[i], batch_first=True,bidirectional=True))
            #self.batch_norms_lstm.append(nn.BatchNorm1d(hidden_sizes[i]))

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(hidden_sizes[-1]*2, n_hidden_nodes[0]))
        self.batch_norms_fc.append(nn.BatchNorm1d(n_hidden_nodes[0]))
        for i in range(1, n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_nodes[i - 1], n_hidden_nodes[i]))
            self.batch_norms_fc.append(nn.BatchNorm1d(n_hidden_nodes[i]))

        self.out = nn.Linear(n_hidden_nodes[-1], num_classes)

    def forward(self, x):
        h, _ = self.lstm[0](x)
        h = F.relu(h) if self.activation == 'relu' else torch.sigmoid(h)
        h = self.dropouts[0](h)

        for i in range(1, len(self.lstm)):
            h, _ = self.lstm[i](h)
            h = F.relu(h) if self.activation == 'relu' else torch.sigmoid(h)
            h = self.dropouts[i](h)
        for i in range(len(self.hidden_layers)):
            h = self.hidden_layers[i](h)
            h = self.batch_norms_fc[i](h)
            h = F.relu(h) if self.activation == 'relu' else torch.sigmoid(h)
            h = self.dropouts[i + len(self.lstm)](h)

        return F.log_softmax(self.out(h), dim=1)

def train(epoch, model, train_loader, loss_function, optimizer, device):
    model.train()
    correct, total_loss = 0, 0
    num_batches = len(train_loader)

    for batch in train_loader:
        data, _, target = batch
        target = target[:, 0]  # Chỉ lấy nhãn đầu tiên nếu target có nhiều chiều
        data, target = data.float().to(device), target.long().to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch}	Loss: {total_loss / num_batches:.6f} Accuracy: {accuracy:.3f}%')


def validate(model, loss_function, validation_loader, device):
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for data, _, target in validation_loader:
            target = target[:, 0]  # Chỉ lấy nhãn đầu tiên nếu target có nhiều chiều
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            val_loss += loss_function(output, target).item()
            correct += output.argmax(dim=1).eq(target).sum().item()

    val_loss /= len(validation_loader)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(
        f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.3f}%)\n')
    return val_loss, accuracy


def DataPreprocessing(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    train_data = load_from_json("train_data.json")
    valid_data = load_from_json("valid_data.json")
    test_data = load_from_json("test_data.json")

    train_dataset = NERDataset(train_data)
    valid_dataset = NERDataset(valid_data)
    test_dataset = NERDataset(test_data)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=DataPreprocessing)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=DataPreprocessing)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=DataPreprocessing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BiLSTM(128, [128,256,512], 3, 2, [512,256], "relu", 25, [0.1,0.2,0.3,0.3,0.3], 0.0001, 5, 10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    for e in range(1, model.epoch + 1):
        train(e, model, train_loader, loss_function, optimizer, device)
        val_loss, _ = validate(model, loss_function, valid_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model.patience:
                print("Early stopping triggered")
                break

    validate(model, loss_function, test_loader, device)


if __name__ == "__main__":
    main()