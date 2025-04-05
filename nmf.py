from tdm_generator import build_matrix
import numpy as np
import sys

def nmf(V, rank, max_iterations=1000, tol = 1e-4):
    np.random.seed(42)
    n_docs, n_words = V.shape

    W = np.random.rand(n_docs, rank)
    H = np.random.rand(rank, n_words)

    prev_error = np.inf
    for _ in range(max_iterations):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)
        error = np.linalg.norm(V - W @ H)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    return W, H

def find_optimal_rank(V, rank_range=(2, 20), max_iterations=100):
    errors = []
    ranks = list(range(rank_range[0], rank_range[1] + 1))

    for rank in ranks:
        W, H = nmf(V, rank=rank, max_iterations=max_iterations)
        error = np.linalg.norm(V - W @ H)
        errors.append(error)
    diffs = np.diff(errors)
    optimal_rank = ranks[np.argmin(diffs) + 1]

    return optimal_rank

def get_main_topic(W, H, vocabulary, top_n=20):
    main_topics = []
    
    for doc_idx in range(W.shape[0]):  
        main_topic_idx = np.argmax(W[doc_idx])
        top_words_idx = np.argsort(H[main_topic_idx])[-top_n:]
        top_words = [vocabulary[i] for i in reversed(top_words_idx)]
        main_topics.append(" ".join(top_words))
    
    return main_topics


def main():
    if len(sys.argv) < 1:
        print("Usage: python3 nmf.py")
        return

    input_folder = "data/documents"
    stop_words = "data/stop_words/stopwords.txt"
    name_path = "data/stop_words/names.txt"
    
    doc_names, vocabulary, term_doc_matrix = build_matrix(input_folder, stop_words, name_path)

    rank = find_optimal_rank(term_doc_matrix, rank_range=(2, 20))
    print(rank)
    W, H = nmf(term_doc_matrix, rank=rank, )
    topics = get_main_topic(W, H, vocabulary)

    for doc, topic in zip(doc_names, topics):
        print(f"{doc}: {topic}")

if __name__ == "__main__":
    main()