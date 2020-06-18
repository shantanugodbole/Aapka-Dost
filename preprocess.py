import nltk
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



# a = "My name is Shantanu Godbole."
# token_a = tokenization(a)
# # lower_a = lowercase(token_a)
# stemmed_a = stemmed_output(token_a)
# refined_a = refineWords(stemmed_a)
# print(token_a)
# print(stemmed_a)
# # print(stemmed_a)
# print(refined_a)
