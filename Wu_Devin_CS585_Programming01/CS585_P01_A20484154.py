import os
import re
import sys
import time

if len(sys.argv) != 4:
    print("ERROR: Not enough or too many input arguments.")
    exit()

script_name = sys.argv[0]
# print("\n Script name = ", script_name)
 
k = int(sys.argv[1])
# print("\n k = ", k)

if k < 1:
    k = 5

train_file_name = sys.argv[2]
# print("\n train.txt = ", train_file_name)

test_file_name = sys.argv[3]
# print("\n test.txt = ", test_file_name)

if not os.path.exists("./" + train_file_name):
    print(train_file_name, " not found in directory")
    exit()
elif not os.path.exists("./" + test_file_name):
    print(test_file_name, " not found in directory")
    exit()


# read file name and store text into array
def text_to_word_array(filename):
    
    with open(filename, 'r') as file:
        text = file.read()
    
    # remove punctuation using regular expressions
    clean_text = re.sub(r'[^\w\s]', '', text)
    
    # split the text into words and remove any extra whitespace
    words = clean_text.split()
    
    return words

train_array = text_to_word_array(train_file_name)
test_array = text_to_word_array(test_file_name)


# break up each word in the train array and append stop token
train_array = [(list(word) + ["_"]) for word in train_array]
test_array = [(list(word) + ["_"]) for word in test_array]


# train_dataframe = pandas.read_csv(train_file_name)
# test_dataframe = pandas.read_csv(test_file_name)

# train_array = train_dataframe[0].tolist()
# test_array = test_dataframe[0].tolist()

# print("train array = ", train_array)
# print("test array = ", test_array)

# initial vocab
vocabulary = [chr(x) for x in range(ord('A'), ord('Z') + 1)] + [chr(x) for x in range(ord('a'), ord('z') + 1)] + ["_"]

# print("initial vocab = ", vocabulary)



# BPE LEARNER ON THE TRAIN FILE HERE
# perform k merges on train array and return the new vocabulary to be added
train_time = 0
token_time = 0
vocabulary_to_be_added = []
for _ in range(k):
    time_start = time.time()
    byte_pair_frequency_dict = {}

    # gather byte pair frequencies
    for word in train_array:
        for i in range(len(word)-1):
            byte_pair = word[i] + word[i+1]

            if byte_pair in byte_pair_frequency_dict:
                byte_pair_frequency_dict[byte_pair] += 1
            else:
                byte_pair_frequency_dict[byte_pair] = 1


    # add most frequent byte pair to vocab
    most_frequent_byte_pair = max(byte_pair_frequency_dict, key=byte_pair_frequency_dict.get)
    vocabulary_to_be_added.append(most_frequent_byte_pair)

    # print("byte pair frequencies at ", k, " = ", byte_pair_frequency_dict)
    # print("most frequenct byte pair at ", k, " = ", max(byte_pair_frequency_dict, key=byte_pair_frequency_dict.get))


    # merge most frequent byte pair
    for word in train_array:
        len_word = len(word)-1
        i = 0
        while i < len_word:
            # print(len_word)
            # print("current letter = ", word[i])
            # print("next letter = ", word[i+1])
            byte_pair = word[i] + word[i+1]
            if byte_pair == most_frequent_byte_pair:
                word[i] = most_frequent_byte_pair
                del word[i+1]
                len_word -= 1
            i += 1
    time_end = time.time()
    train_time += time_end - time_start
    # print("merged train array at ", k, " = ", train_array)

    # BPE SEGMENTER AND TOKENIZING TEST FILE HERE
    time_start = time.time()
    for word in test_array:
        len_word = len(word)-1
        i = 0
        while i < len_word:
            byte_pair = word[i] + word[i+1]
            if byte_pair == most_frequent_byte_pair:
                word[i] = most_frequent_byte_pair
                del word[i+1]
                len_word -= 1
            i += 1
    time_end = time.time()
    token_time += time_end - time_start

# print("vocab to be added = ", vocabulary_to_be_added)

# print("merged train array = ", train_array)

# cleaning up stop tokens from test result
for i in range(len(test_array)):
    word = test_array[i]
    for ii in range(len(word)):
        token = test_array[i][ii]
        if token == "_":
            # print("DELETING ", token, " IN WORD ", word)
            del test_array[i][ii]
        else:
            test_array[i][ii] = token.replace("_", "")


    # if word == "_":
    #     del test_array[i]
    # else:
    #     for ii in range(len(word)):
    #         token = word[ii]
    #         token = token.replace("_", "")
# for word in test_array:
#     for token in word:
#         if token == "_":
#             print("DELETING ", token, " IN WORD ", word)
#             del token
#         else:
#             token = token.replace("_", "")

final_vocab = vocabulary + vocabulary_to_be_added
vocab_file = open("CS585_P01_A20484154_VOCAB.txt", "w")
# if len(vocabulary_to_be_added) > 20:
#     for i in range(20):
#         vocab_file.write(vocabulary_to_be_added[i])
#     vocab_file.write("Tokenized text is longer than 20 tokens")
# else:
#     for token in vocabulary_to_be_added:
#         vocab_file.write(token)
for token in final_vocab:
        vocab_file.write(token + "\n")
vocab_file.close()

result_file = open("CS585_P01_A20484154_RESULT.txt", "w")
for word in test_array:
    for token in word:
        result_file.write(token + " ")
    result_file.write("  ")
result_file.close()

# print("final tokenized test file = ", test_array)
# print("training time = ", round(train_time, 5), " seconds")
# print("tokenizing time = ", round(token_time, 5), " seconds")

# OUTPUT
print("Wu, Devin, A20484154 solution:")
print("Number of merges: ", k)
print("Training file name: ", train_file_name)
print("Test file name: ", test_file_name)
print("\nTraining time: ", round(train_time, 5), " seconds")
print("Tokenization time: ", round(token_time, 5), " seconds")
print("\nTokenization result: ")
if len(vocabulary_to_be_added) > 20:
    for i in range(20):
        print(vocabulary_to_be_added[i])
    print("Tokenized text is longer than 20 tokens")
else:
    for token in vocabulary_to_be_added:
        print(token)
