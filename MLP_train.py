import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torch import device
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets

def train(epoch, model, train_loader, optimizer, log_interval=100):
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1)# index của cái có xác suất log cao nhất
            correct += pred.eq(target).sum().item()
            accuracy = 100. * correct / len(train_loader.dataset) # 100. -> float
            loss = torch.nn.functional.nll_loss(output, target) # negative log-likelihood
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: \tLoss: {:.6f} Accuracy (%): {:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), accuracy))
def validate(loss_vector, accuracy_vector, model, validation_loader):
        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            output = model(data)
            val_loss += torch.nn.functional.nll_loss(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        val_loss /= len(validation_loader)
        loss_vector.append(val_loss)
        accuracy = 100.*correct/len(validation_loader.dataset)
        accuracy_vector.append(accuracy)
        print('\nValidation set: Average loss: {:.4f}, Accuracy (%): {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy))
def load_data():
    transform = transforms.Compose(
           [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                        shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, validation_loader