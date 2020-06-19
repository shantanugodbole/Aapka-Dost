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

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        all_words.extend(w)
        matcher.append((w, tag))

all_words = refineWords(all_words)
all_words = stemmed_output(all_words)

#Unique words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Creating training data
X_train = []
y_train = []

for (sentence, tag) in matcher:
    bag_of_words = bagofWords(sentence, all_words)
    X_train.append(bag_of_words)

    label = tag.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = torch.from_numpy(y_train)
y_train = torch.tensor(y_train, dtype=torch.long)


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
train_data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
epochs = 1000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

#loss and optimization
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    for(words, labels) in train_data_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward propogation
        outputs = model(words)
        loss = criteria(outputs, labels)

        #backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0 :
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

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