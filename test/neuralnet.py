import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class Train_Test():
    def __init__(self):
        pass

    def accuracy(self,outputs, labels):
        _, preds = torch.max(outputs, 1)
        return torch.sum(preds == labels).item() / len(labels)


    def train( self,model, device, train_loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += self.accuracy(outputs, labels)
            if (i + 1) % 200 == 0:
                print(
                    f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
                running_loss = 0.0
                running_acc = 0.0


    def test(self,model, device, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += self.accuracy(outputs, labels)
        print(
            f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}')