import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class MLP(torch.nn.Module):
    def __init__(self,input_shape, batch_size, epoch, n_hidden_nodes, n_hidden_layers, activation,drop_out=0, learning_rate = 0, patience =10):
        super(MLP, self).__init__()
        self.patience= patience
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        if learning_rate == 0:
            learning_rate = 0.005
        self.learning_rate = learning_rate
        # set up layers n_hidden_nodes, layers, activation, keep_rate, drop, out
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layer_drop = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(input_shape[0]*input_shape[1]*input_shape[2],n_hidden_nodes[0]))
        self.hidden_layer_drop.append(torch.nn.Dropout(drop_out))
        for i in range(1, n_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(n_hidden_nodes[i-1], n_hidden_nodes[i]))
            self.hidden_layer_drop.append(torch.nn.Dropout(drop_out))
        self.out = torch.nn.Linear(n_hidden_nodes[n_hidden_layers-1], 10)

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        if self.activation == 'sigmoid':
            sigmoid = torch.nn.Sigmoid()
            for i in range(self.n_hidden_layers):
                x = sigmoid(self.hidden_layers[i](x))
                x = self.hidden_layer_drop[i](x)
        elif self.activation == 'relu':
            for i in range(self.n_hidden_layers):
                x = torch.nn.functional.relu(self.hidden_layers[i](x))
                x = self.hidden_layer_drop[i](x)
        return torch.nn.functional.log_softmax(self.out(x))

def train(epoch, model, train_loader,loss_function, optimizer):
        model.train()
        correct = 0
        total_loss = 0
        num_batches = len(train_loader)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            loss = loss_function(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        accuracy = 100. * correct / len(train_loader.dataset)
        average_loss = total_loss / num_batches
        print('Train Epoch: {}\tLoss: {:.6f} Accuracy: {:.3f}%'.format(epoch, average_loss, accuracy))


def validate(model, loss_function, validation_loader):
        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            output = model(data)
            val_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        val_loss /= len(validation_loader)
        accuracy = 100.*correct/len(validation_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy (%): {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy))
        return val_loss, accuracy
def main():
    batch_size = 4
    best_val_loss = float('inf')
    patience_counter= 0
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               pin_memory=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                    shuffle=False, num_workers=0, pin_memory=False)
    model = MLP([32, 32, 3],batch_size,50, [512, 256, 128], 3, 'relu', 0.3,0.001)
    for e in range(1, model.epoch+1):
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
        train(e, model, train_loader, loss_function, optimizer)
        val_loss, val_acc = validate(model, loss_function, validation_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model.patience:
                print("Early stopping triggered")
                break
main()
