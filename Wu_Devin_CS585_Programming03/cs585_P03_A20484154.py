import os
import re
import sys
import pandas as pd
import numpy as np
import nltk
from nltk import RegexpTokenizer

pd.set_option('mode.chained_assignment', None)

tokenizer = RegexpTokenizer(r'\w+')

if len(sys.argv) != 3:
    print("ERROR: Not enough or too many input arguments.")
    exit()

algo = int(sys.argv[1])
if (algo != 0) and (algo != 1):
    print("Choose algorithm of naive bayes (0) or logistic regression (1)")
    exit()

 
train_size = int(sys.argv[2])
if train_size < 50 or train_size > 80:
    print("Choose a train size between 50 and 80")
    exit()

# algo = 0
# train_size = 80

print("Wu, Devin, A20484154 Solution:")
print("Training set size", train_size, "%")
classifier = "Naive Bayes" if algo == 0 else "Logistic Regression"
print("Classifier Type:", classifier)
print("")
print("Training classifier...")
print("Testing classifier...")
print("")

true_csv = pd.read_csv('True A20484154.csv')
fake_csv = pd.read_csv('Fake A20484154.csv')

# DATA CLEANUP

true_csv = true_csv.drop(columns=['title', 'subject', 'date'])
fake_csv = fake_csv.drop(columns=['title', 'subject', 'date'])

true_csv = np.array(true_csv['text'])
fake_csv = np.array(fake_csv['text'])


for i in range(len(true_csv)):
    true_csv[i] = tokenizer.tokenize(true_csv[i].lower())
for i in range(len(fake_csv)):
    fake_csv[i] = tokenizer.tokenize(fake_csv[i].lower())

counter = 0
while counter < len(true_csv):
    if len(true_csv[counter]) >= 400:
        true_csv[counter] = np.array(true_csv[counter][:400])
        counter += 1
    else:
        true_csv = np.delete(true_csv, counter, 0)
counter = 0
while counter < len(fake_csv):
    if len(fake_csv[counter]) >= 400:
        fake_csv[counter] = np.array(fake_csv[counter][:400])
        counter += 1
    else:
        fake_csv = np.delete(fake_csv, counter, 0)

new_arr = np.empty((len(true_csv), 400), dtype='<U50')
for row in range(len(true_csv)):
    for col in range(400):
        new_arr[row][col] = true_csv[row][col]
true_csv = new_arr
new_arr = np.empty((len(fake_csv), 400), dtype='<U50')
for row in range(len(fake_csv)):
    for col in range(400):
        new_arr[row][col] = fake_csv[row][col]
fake_csv = new_arr

# for tokens in true_csv:
#     print(tokens)


# true_csv = np.reshape(len(true_csv), 400)
# fake_csv = np.reshape(len(fake_csv), 400)

# np.arange((len(true_csv)* 400)).

# print(true_csv.ndim)
# print(true_csv.shape)
# print(true_csv.size)

# for tokens in true_csv:
#     if len(tokens) != 400:
#         print(tokens)
# print(len(true_csv), len(true_csv[0]))


# print(len(true_csv[5000]))




# true_csv = np.array([row for row in true_csv if len(row)<400])



# for tokens in true_csv:
#     print(len(tokens))
# print(true_csv.shape)



# for text in true_csv:
#     text = tokenizer.tokenize(text.lower())
# for text in fake_csv:
#     text = tokenizer.tokenize(text.lower())

true_original_len = len(true_csv)
fake_original_len = len(fake_csv)

    # TAKING LAST 20% FOR TEST SET

true_test_csv = true_csv[int(true_original_len*0.8):]
fake_test_csv = fake_csv[int(fake_original_len*0.8):]

    # TAKING FIRST CHOSEN TRAIN SIZE % FOR TRAIN SET
true_csv = true_csv[:int(true_original_len*train_size*0.1)]
fake_csv = fake_csv[:int(true_original_len*train_size*0.1)]

# for text in true_csv['text']:  
#     print(tokenizer.tokenize(text))

all_unique_words = set()
for tokens in true_csv:
    for token in tokens:
        all_unique_words.add(token)
for tokens in fake_csv:
    for token in tokens:
        all_unique_words.add(token)

count_all_unique_words = len(all_unique_words)
# print("count all unique words = ", count_all_unique_words)

true_positives = 0      # true positive = correctly says true news is true
false_postives = 0      # false positive = incorrectly says fake news is true
true_negatives = 0      # true negative = correctly says fake news is fake
false_negatives = 0     # false negative = incorrectly says true news is fake

# NAIVE BAYES
if algo == 0:
    true_prior_prob = np.log(len(true_csv) / (len(true_csv) + len(fake_csv)))
    fake_prior_prob = np.log(len(fake_csv) / (len(true_csv) + len(fake_csv)))

    # print("true prior = ", true_prior_prob)
    # print("fake prior = ", fake_prior_prob)

    true_dict = dict()
    fake_dict = dict()

    for tokens in true_csv:
        for token in tokens:
            if token not in true_dict:
                true_dict[token] = 1
            else:
                true_dict[token] += 1

    for tokens in fake_csv:
        for token in tokens:
            if token not in fake_dict:
                fake_dict[token] = 1
            else:
                fake_dict[token] += 1


    # NAIVE BAYES CLASSIFY FUNCTION

    def naive_classify(sentence):
        true_prob = true_prior_prob
        fake_prob = fake_prior_prob

        for token in sentence:
            if token in true_dict:
                true_prob += np.log((true_dict[token]+1)/ (len(true_dict)+count_all_unique_words))
            else:
                true_prob += np.log(1 / (len(true_dict)+count_all_unique_words)) # add 1 smoothing
            if token in fake_dict:
                fake_prob += np.log((fake_dict[token]+1) / (len(fake_dict)+count_all_unique_words))
            else:
                fake_prob += np.log(1 / (len(fake_dict)+count_all_unique_words)) # add 1 smoothing
        # np.exp(true_prob), np.exp(fake_prob) # returning to linear space resulted in too small of a number
        return true_prob, fake_prob

    # print(true_csv[0])
    # print("all count = ", count_all_unique_words)
    # test_sentence = "the quick brown fox jumps over the lazy dog"
    # true_prob, fake_prob = naive_classify(test_sentence)
    # print("true_prob = ", true_prob)
    # print("fake prob = ", fake_prob)


    # NAIVE BAYES TEST RESULTS


    # y_pred = []

    for text in true_test_csv:
        true, fake = naive_classify(text)
        if true > fake:
            true_positives += 1
            # y_pred.append(1)
        else:
            false_negatives += 1
            # y_pred.append(-1)
    for text in fake_test_csv:
        true, fake = naive_classify(text)
        if true > fake:
            false_postives += 1
            # y_pred.append(-1)
        else:
            true_negatives += 1
            # y_pred.append(0)
    
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import roc_curve, auc

    # y_test = ([1] * len(true_test_csv)) + ([0] * len(fake_test_csv))
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.show()

    # print("TP, FP, TN, FN = ", true_positives, false_postives, true_negatives, false_negatives)

# LOGISTIC REGRESSION
if algo == 1:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    # USING TF IDF VECTORS TO CALCULATE

    term_count_in_all_docs = dict() # count of each token in every document
    for tokens in true_csv:
            for token in tokens:
                if token not in term_count_in_all_docs:
                    term_count_in_all_docs[token] = 1
                else:
                    term_count_in_all_docs[token] += 1
    for tokens in fake_csv:
            for token in tokens:
                if token not in term_count_in_all_docs:
                    term_count_in_all_docs[token] = 1
                else:
                    term_count_in_all_docs[token] += 1

    # print("term count = ", term_count_in_all_docs)
    # print("all unique words = ", count_all_unique_words)

    def tokens_to_tfidf(tokens):
        tfidf_tokens = []
        tokens_count = dict()
        for token in tokens:
            if token not in tokens_count:
                tokens_count[token] = 1
            else:
                tokens_count[token] += 1
        
        # calculate tfidf
        for token in tokens:
            if token not in term_count_in_all_docs:
                tfidf = (tokens_count[token] + 1) * count_all_unique_words
            else:
                tfidf = (tokens_count[token] + 1) * (term_count_in_all_docs[token] + count_all_unique_words) # add 1 smoothing
            # print("tfidf of ", token, " = ", tfidf)
            # print("type of tfidf = ", type(tfidf))
            tfidf_tokens.append(tfidf)

        return tfidf_tokens




    # PREPARING DATA FOR LOGISTIC REGRESSION



    x_train = np.concatenate([true_csv, fake_csv], axis=0)
    y_train = ([1] * len(true_csv)) + ([0] * len(fake_csv))
    x_test = np.concatenate([true_test_csv, fake_test_csv], axis=0)
    y_test = ([1] * len(true_test_csv)) + ([0] * len(fake_test_csv))



    new_arr = np.empty((len(x_train), 400), dtype='int')
    for i in range(len(x_train)):
        # print("tfidf value = ", tokens_to_tfidf(x_train[i]))
        new_arr[i] = tokens_to_tfidf(x_train[i])
    x_train = new_arr
    new_arr = np.empty((len(x_test), 400), dtype='int')
    for i in range(len(x_test)):
        # print("tfidf value = ", tokens_to_tfidf(x_test[i]))
        new_arr[i] = tokens_to_tfidf(x_test[i])
    x_test = new_arr

    # print("type of true csv = ", type(true_csv[0][0]))
    # print("type of x train = ", type(x_train[0][0]))
    # print("type of true test = ", type(true_test_csv[0][0]))
    # print("type of x test = ", type(x_test[0][0]))


    # x_train = np.reshape(len(x_train), 400)
    # x_test = np.reshape(len(x_test), 400)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # print("x train shape = ", x_train.shape)
    # print("y train shape = ", y_train.shape)
    # print("x test shape = ", x_test.shape)
    # print("y test shape = ", y_test.shape)


    # print("true_csv shape = ", true_csv.shape)
    # print("fake_csv shape = ", fake_csv.shape)
    # print("true_test_csv shape = ", true_test_csv.shape)
    # print("fake_test_csv shape = ", fake_test_csv.shape)

    # for tokens in x_test:
    #     print(type(tokens))
    #     print(type(tokens[0]))
    #     print(len(tokens))

    # for i in range(10):
    #     print(x_train[i])


    # for classification in y_train:
    #     print(classification)
    #     print(type(classification))
        # print(len(classification))








    # TRAINING LOGISTRIC REGRESSION CLASSIFIER

    logreg_model = LogisticRegression(max_iter=100)

    logreg_model.fit(x_train, y_train)

    y_pred = logreg_model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)

    true_negatives, false_postives, false_negatives, true_positives = cm.ravel()

    # import matplotlib.pyplot as plt
    # from sklearn.metrics import roc_curve, auc

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.show()

    # print("x train len = ", len(x_train))
    # print("y train len = ", len(y_train))
    # print("x test len = ", len(x_test))
    # print("y test len = ", len(y_test))




sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_postives)
precision = true_positives / (true_positives + false_postives)
negative_predictive_value = true_negatives / (true_negatives + false_negatives)
accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_postives + false_negatives)
f_score = (2 * precision * sensitivity) / (precision + sensitivity)

# PRINT RESULTS

print("Test results / metrics:")
print("Number of true positives:", true_positives)
print("Number of true negatives:", true_negatives)
print("Number of false positives:", false_postives)
print("Number of false negatives:", false_negatives)
print("Sensitivity (recall):", np.round(sensitivity, decimals=4))
print("Specificity:", np.round(specificity, decimals=4))
print("Precision:", np.round(precision, decimals=4))
print("Negative predictive value:", np.round(negative_predictive_value, decimals=4))
print("Accuracy:", np.round(accuracy, decimals=4))
print("F-score:", np.round(f_score, decimals=4))
print("")
def user_input():
    sentence = input("Enter your sentence/document:")
    print("")
    print("Sentence/document S:", sentence)
    print("")

    sentence = tokenizer.tokenize(sentence.lower())
    prediction = "True"
    if algo == 0:
        true_prob, fake_prob = naive_classify(sentence)
        if true_prob > fake_prob:
            prediction = "True"
        else:
            prediction = "Fake"
    else:
        tokens = [tokens_to_tfidf(sentence)]
        if len(tokens[0]) > 400:
            tokens[0] = tokens[0][:400]
        else:
            for _ in range(400 - len(tokens[0])):
                tokens[0].append(count_all_unique_words)
        prediction = logreg_model.predict(tokens)

        if prediction[0] == 1:
            prediction = "True"
        else:
            prediction = "Fake"


    print("Was classified as ", prediction, ".")
    if algo == 0:
        print("P(True | S) =", np.exp(true_prob))
        print("P(Fake | S) =", np.exp(fake_prob))
    print("")

user_input()

cont = 'y'
while cont.lower() == 'y':
    cont = input("Do you want to enter another sentence [Y/N]?")
    if cont.lower() == 'y':
        user_input()

    