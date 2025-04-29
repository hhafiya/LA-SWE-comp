import numpy as np
import sys
import json
from train import build_matrix, reduce, softmax

# get path to new book from command line
if (len(sys.argv) <= 1):
    print("Please enter the path to a directory with books.")
    exit(1)
else:
    path = sys.argv[1]


# load the necessary data
try:
    with open('data/train_result/vocabulary.txt', 'r') as f:
        vocabulary = f.read().splitlines()

    with open('data/train_result/index_to_label.json', 'r') as f:
        index_to_label = json.load(f)
    W = np.load("data/train_result/W.npy")
    weights = np.load("data/train_result/weights.npy")
    bias = np.load("data/train_result/bias.npy")
    
except:
    print("Error opening the necessary data from training. Try rerunning train.py.")
    exit(1)



def predict(X, weights, bias, book_titles, index_to_label):
    predictions = []
    for xi, title in zip(X, book_titles):
        z = np.dot(xi, weights) + bias
        probs = softmax(z)
        probs_sum = 0 # aim to predict with at least 0.75 probability
        predictions_for_book = []
        probs_indeces = np.argsort(probs)[::-1] #indeces of sorted probabilities
        i = 0
        # maximum 3 predicted labels
        while (probs_sum < 0.75 and i < 3):
            pred_idx = str(probs_indeces[i])
            predicted_label = index_to_label[pred_idx]
            predicted_prob = probs[int(pred_idx)]
            probs_sum += predicted_prob
            predictions_for_book.append((predicted_label, predicted_prob))
            i += 1
        
        predictions.append((title, predictions_for_book))
        

    return predictions





stopwords_path = "data/stop_words/stopwords.txt"
name_path = "data/stop_words/names.txt"
doc_names_test, _, X_test = build_matrix(path, stopwords_path, name_path, vocabulary)


X_test_reduced = np.dot(X_test, W)



predictions = predictions_train = predict(X_test_reduced, weights, bias, doc_names_test, index_to_label)

print("\n== Prediction on provided data ==")

for title, prediction in predictions:
    print(f"Book: {title}, Predicted Genre(s): {', '.join(f'{label} Probability: {prob:.4f}' for label, prob in prediction)}")