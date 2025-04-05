from collections import Counter
from pathlib import Path
import numpy as np
import re
import string

def clean_text(line):
    line = line.lower()
    punctuation_allowed = "'-"
    punctuation_to_remove = ''.join(c for c in string.punctuation if c not in punctuation_allowed)
    line = line.translate(str.maketrans('', '', punctuation_to_remove))
    line = ' '.join(re.findall(r"\b[a-zA-Z]+(?:['-][a-zA-Z]+)*\b", line))
    return line


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
            word_counter.update(filtered)
    return word_counter


def build_matrix(folder_path, stopwords_path, name_path):
    matrix = {}
    vocabulary = set()

    for file_path in Path(folder_path).glob("*.txt"):
        word_counts = clean_text_count(file_path, stopwords_path, name_path)
        matrix[file_path.name] = word_counts
        vocabulary.update(word_counts.keys())

    vocabulary = sorted(vocabulary)
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    doc_names = list(matrix.keys())
    
    term_doc_matrix = np.zeros((len(doc_names), len(vocabulary)), dtype=int)

    for i, doc in enumerate(doc_names):
        for word, count in matrix[doc].items():
            j = vocab_index[word]
            term_doc_matrix[i][j] = count

    return doc_names, vocabulary, term_doc_matrix