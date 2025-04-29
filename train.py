from collections import Counter
from pathlib import Path
import numpy as np
import re
import string
import json

def clean_text(line):
    line = line.lower()
    punctuation_allowed = "'-"
    punctuation_to_remove = ''.join(c for c in string.punctuation if c not in punctuation_allowed)
    line = line.translate(str.maketrans('', '', punctuation_to_remove))
    line = re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", line)
    line = [word[:-2] if word.endswith("'s") else word for word in line]
    return ' '.join(line)


def clean_text_count(file_path, stopwords_path, name_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set(word.strip().lower() for word in f.readlines())

    with open(name_path, 'r', encoding='utf-8') as f:
        names = set(word.strip().lower() for word in f.readlines())

    word_counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = clean_text(line)
            words = cleaned_line.split()
            filtered = [word for word in words if word not in stopwords and word not in names]
            filtered = [word for word in words if word not in stopwords]
            word_counter.update(filtered)
    return word_counter


def build_matrix(folder_path, stopwords_path, name_path, vocabulary=None):
    raw_counts = {}
    for file_path in Path(folder_path).glob("*.txt"):
        wc = clean_text_count(file_path, stopwords_path, name_path)
        raw_counts[file_path.name] = wc

    doc_names = sorted(raw_counts.keys())

    if vocabulary is None:
        vocab_set = set()
        for wc in raw_counts.values():
            vocab_set.update(wc.keys())
        vocabulary = sorted(vocab_set)

    vocab_index = {w:i for i,w in enumerate(vocabulary)}

    M = np.zeros((len(doc_names), len(vocabulary)), dtype=int)

    for i,doc in enumerate(doc_names):
        wc = raw_counts[doc]
        for word,count in wc.items():
            j = vocab_index.get(word)
            if j is not None:
                M[i,j] = count

    return doc_names, vocabulary, M

folder_path = "data/train"
stopwords_path = "data/stop_words/stopwords.txt"
name_path = "data/stop_words/names.txt"

doc_names, vocabulary, X_train = build_matrix(folder_path, stopwords_path, name_path)

# store vocabulary in .json file to use for prediction
with open('data/train_result/vocabulary.txt', 'w') as f:
    f.write('\n'.join(vocabulary))


def svd(matrix):
    transpose = np.transpose(matrix)
    work_matrix = np.dot(matrix, transpose)

    eigenvalues, left_sing_matrix_U = np.linalg.eigh(work_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    left_sing_matrix_U = left_sing_matrix_U[:, idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    sigma_E = np.diag(singular_values)

    right_sing_matrix_V = []
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:
            v_i = np.dot(transpose, left_sing_matrix_U[:, i]) / singular_values[i]
        else:
            v_i = np.zeros(matrix.shape[1])
        right_sing_matrix_V.append(v_i)
    right_sing_matrix_V = np.column_stack(right_sing_matrix_V)

    return left_sing_matrix_U, sigma_E, right_sing_matrix_V

U, Sigma, V = svd(X_train)


k = 300
def reduce(V, k):
    reduced = V[:, :k]
    return reduced
W = reduce(V, k)
np.save("data/train_result/W.npy", W)
X_train_reduced = np.dot(X_train, W)


# IMPORTANT: when changing the training set, update doc_labels
doc_labels = ["fantasy"] * 5 + ["mystery"] * 5 + ["science fiction"] * 5 + ["horror"] * 5 + ["romance"] * 5 + ["adventure"] * 5  + ["self help"] * 5 + ["textbook"] * 5

unique_labels = sorted(set(doc_labels))
label_to_index = {l: i for i, l in enumerate(unique_labels)}
index_to_label = {i: l for l, i in label_to_index.items()}

with open("data/train_result/index_to_label.json", "w") as f:
    json.dump(index_to_label, f)

y = np.array([label_to_index[l] for l in doc_labels])
y_onehot = np.eye(len(unique_labels))[y]

n_features = X_train_reduced.shape[1]
n_classes = len(unique_labels)


def softmax(logits):
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits_shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def gradient_descent(X, y_onehot, W, b, lr, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for xi, yi in zip(X, y_onehot):

            z = np.dot(xi, W) + b
            probs = softmax(z)

            loss = -np.sum(yi * np.log(probs + 1e-8))
            total_loss += loss

            dz = probs - yi
            dW = np.outer(xi, dz)
            db = dz

            reg_lambda = 1e-2
            dW += reg_lambda * W

            W -= lr * dW
            b -= lr * db

    return W, b


weights = np.random.randn(n_features, n_classes) * 0.01
bias = np.zeros(n_classes)
learning_rate = 0.05
epochs = 500
weights, bias = gradient_descent(X_train_reduced, y_onehot, weights, bias, learning_rate, epochs)

np.save("data/train_result/weights.npy", weights)
np.save("data/train_result/bias.npy", bias)
