import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class MLP(torch.nn.Module):
    def __init__(self,input_shape, batch_size, epoch, n_hidden_nodes, n_hidden_layers, activation,drop_out=0, learning_rate = 0):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        if learning_rate == 0:
            learning_rate = 0.5
        # set up layers n_hidden_nodes, layers, activation, keep_rate, drop, out
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layer_drop = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(input_shape[1]*input_shape[2]*input_shape[3],n_hidden_nodes[0]))
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

def train(epoch, model, train_loader,loss_function, optimizer, log_interval=100):
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            accuracy = 100. * correct / len(train_loader.dataset)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.3f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), accuracy))

def validate(model, validation_loader):
        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            output = model(data)
            val_loss += torch.nn.functional.nll_loss(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        val_loss /= len(validation_loader)
        accuracy = 100.*correct/len(validation_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy (%): {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy))
def main():
    batch_size = 4
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               pin_memory=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                    shuffle=False, num_workers=0, pin_memory=False)
    model = MLP([10000, 32, 32, 3],batch_size,30, [64, 32], 2, 'sigmoid', 0.3,[.00001, 0.0001, 0.001, 0.01, 0.1])
    for e in range(1, model.epoch+1):
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        train(e, model, train_loader, loss_function, optimizer)
        validate(model, validation_loader)
main()