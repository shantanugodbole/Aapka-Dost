import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from preprocess import tokenization, stemmed_output, refineWords, bagofWords
from model import NeuralNetwork

with open('intents.json', 'r') as file:
    intents = json.load(file)

# Create a list of all words for bag of words model.
all_words = []
tags = []
matcher = []

# Extracting tags and sentences from intents
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        all_words.extend(w)
        matcher.append((w, tag))

all_words = refineWords(all_words)
all_words = stemmed_output(all_words)

# Unique words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Creating training data
X_train = []
y_train = []

for (sentence, tag) in matcher:
    bag_of_words = bagofWords(sentence, all_words)
    X_train.append(bag_of_words)

    label = tags.index(tag)
    y_train.append(label)

# Creating training dataset
X_train = np.array(X_train)
y_train = np.array(y_train)

# Checking system config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Converting list of labels into tensors, for NN model
y_train = torch.tensor(y_train, dtype=torch.long, device = device) 

# HYPERPARAMETERS
input_size = len(X_train[0])
hidden_size = 10
batch_size = 10
output_size = len(tags)
learning_rate = 0.0009
epochs = 1000

class ChitterChatter(Dataset):
    def __init__(self):
        self.n_examples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_examples


dataset = ChitterChatter()
train_data_loader = DataLoader(dataset, batch_size, shuffle=True)

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

#loss and optimization
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for(words, labels) in train_data_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward propogation
        outputs = model(words)
        loss = criteria(outputs, labels)

        # backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
