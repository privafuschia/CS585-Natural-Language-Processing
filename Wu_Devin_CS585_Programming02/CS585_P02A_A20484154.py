import nltk
from nltk import bigrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import brown


nltk.download('brown')
nltk.download('punkt') #tokenizer

brown_words = brown.words()
brown_bigrams = bigrams(brown_words)
# print(brown_bigrams)
# print(type(brown_bigrams))

# test sentence
# The quick brown fox jumps over the lazy dog.

# START
sentence = input("enter a sentence:\n")
sentence = sentence.lower()
tokens = nltk.word_tokenize(sentence)
# print(type(tokens))
sentence_bigrams = list(bigrams(tokens))

# print("tokens are: ", tokens)
# print("bigrams are: ", sentence_bigrams)


freq_dist = FreqDist(brown_words)
# print(type(freq_dist))
# print(freq_dist)
cond_freq_dist = ConditionalFreqDist(brown_bigrams)

# take each bigram and return probability
def bigram_to_probability(word1, word2):
    freq_word1 = freq_dist[word1]
    if freq_word1 == 0:
        return 0
    
    # prob of word2 given word1
    return cond_freq_dist[word1][word2] / freq_word1

# calculate final probabilties
probability = 1.0
bigram_probabilities_to_print = []
for bigram in sentence_bigrams:
    word1, word2 = bigram
    bigram_prob = bigram_to_probability(word1, word2)
    bigram_probabilities_to_print.append(bigram_prob)
    probability *= bigram_prob

# every sentence will have 1 starting and ending bigram worth 0.25
probability *= 0.125


print("Your sentence, good person =", sentence)
print("The probability of this sentence is =", probability)
for bigram, bigram_prob in zip(sentence_bigrams, bigram_probabilities_to_print):
    print("The probability of bigram,", bigram, "is", bigram_prob)

input("enter to exit\n")