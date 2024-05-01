# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:31:44 2024

@author: TWP
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from utils import mnist_reader
from NN import NN, SGD, Dataloader
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act_func=nn.ReLU()):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act_func = act_func
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.fc3(x)
        import pdb;pdb.set_trace()
        x = self.softmax(x)
        return x
    
input_size = 784
hidden_size = 256
output_size = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


criterion = nn.CrossEntropyLoss()

learning_rate = 0.4
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

data_path = os.path.join('data','fashion')
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

batch_size = 128
train_data = Dataloader([X_train/255, y_train],batch_size=batch_size)

model = Classifier(input_size, hidden_size, output_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for _ in range(20):
    loss_sum = 0
    for batch in tqdm(train_data,total=len(train_data)):
        x,y = batch
        x = torch.from_numpy(x).to(device)
        x = x.float()
        y = torch.from_numpy(y).to(device)
        optimizer.zero_grad()
        loss = criterion(model(x),y)
        loss.backward()
        optimizer.step()
        loss_sum += loss
    print(loss_sum)

test_data = Dataloader([X_test/255, y_test],batch_size=batch_size)
total = 0
correct = 0
with torch.no_grad():
    for batch in tqdm(test_data, total=len(test_data)):
        x,y = batch
        x = torch.from_numpy(x).to(device)
        x = x.float()
        y = torch.from_numpy(y).to(device)
        outputs = model(x)
        predicted_labels = torch.argmax(outputs, dim=1)
        total += batch_size
        correct += (predicted_labels == y).sum().item()
    accuracy = 100 * correct / total
print('\n', accuracy)











