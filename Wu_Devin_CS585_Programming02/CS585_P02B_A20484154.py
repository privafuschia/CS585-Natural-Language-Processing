import nltk
from nltk import bigrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import brown
from nltk.corpus import stopwords

nltk.download('brown')
nltk.download('punkt') # tokenizer
nltk.download('stopwords')

brown_words = brown.words()
brown_bigrams = bigrams(brown_words)
stop_words = stopwords.words('english')
# print("stopwords =", stop_words)
# print(type(stop_words))

# filter corpus for stopwords and punctuation
brown_words = [word.lower() for word in brown_words if word.isalpha() and word.lower() not in stop_words]


freq_dist = FreqDist(brown_words)
cond_freq_dist = ConditionalFreqDist(brown_bigrams)


# take each bigram and return probability
def bigram_to_probability(word1, word2):
    freq_word1 = freq_dist[word1]
    if freq_word1 == 0:
        return 0
    
    # prob of word2 given word1
    return cond_freq_dist[word1][word2] / freq_word1



# START
start_word = input("enter a word/token\n")
start_word = start_word.lower()

# ask again if first word is not corpus
while start_word not in brown_words:
    start_word = input(start_word + " not found in corpus, try again or ENTER nothing to quit\n")
    if start_word == "":
        print("exited")
        exit()

sentence = []
sentence.append(start_word)

word1 = start_word
while True:

    bigram_probs = dict()
    for word2 in brown_words:
        bigram_probs[word2] = bigram_to_probability(word1, word2)
    top_keys = sorted(bigram_probs, key=bigram_probs.get, reverse=True)[:3]

    # for i in range(3):
    #     print(top_probs[i])

    print(word1, "...\n")
    print("Which word should follow:")
    for i in range(3):
        word2, prob = top_keys[i], bigram_probs[top_keys[i]]
        print(f"{i+1}) {word2} P({word1} {word2}) = {prob:.5f}")

    choice = input("4) QUIT\n")


    # choice = int(choice) if choice.isdigit() else choice = 1
    if choice.isdigit():
        choice = int(choice)
    else:
        choice = 1
    
    choice = 1 if choice not in [1,2,3,4] else choice

    if choice == 4:
        print("Your final sentence is: " + ' '.join(sentence))
        exit()
    
    print("you selected", choice)
    print()

    word1 = top_keys[choice-1]
    sentence.append(word1)


    
