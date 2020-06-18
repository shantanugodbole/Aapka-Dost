import json
from preprocess import tokenization, stemmed_output, refineWords

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
