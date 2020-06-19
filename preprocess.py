import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# Tokenization - Converting sentences into words/tokens.
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

# Convert to lowercase
# Stemming - Find base word of different words (like Lemmatization)
def stemmed_output(words):
    result = []
    lower_words = []
    for word in words:
        temp = word.lower()
        lower_words.append(temp)
    for word in lower_words:
        result.append(stemmer.stem(word))
    return result


# Remove punctuation
def refineWords(stemmed):
    for word in stemmed:
        if word == '?' or word == '!' or word == ',' or word =='.':
            stemmed.remove(word)
    return stemmed


def bagofWords(tokenized_sentence, all_words):
    stemmed_sent = stemmed_output(tokenized_sentence)
    bow = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in stemmed_sent:
            bow[index]=1
    
    return bow
