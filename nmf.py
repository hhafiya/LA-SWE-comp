from tdm_generator import build_matrix
import numpy as np
import sys
from transformers import pipeline

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

        if abs(prev_error - error) <= tol:
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

def get_main_topic(W, H, vocabulary, top_k_topics=1, top_n_words=200):
    main_topics = []

    for doc_idx in range(W.shape[0]):
        top_topic_indices = np.argsort(W[doc_idx])[-top_k_topics:]
        words_set = set()

        for topic_idx in reversed(top_topic_indices):
            top_words_idx = np.argsort(H[topic_idx])[-top_n_words:]
            for i in top_words_idx:
                words_set.add(vocabulary[i])

        main_topics.append(" ".join(words_set))

    return main_topics

def create_pseudo_text(words):
    word_list = words.split()
    return f"{'Text is about' + ', '.join(word_list)}"

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

    candidate_labels = [
        "fantasy",
        "romance",
        "science fiction",
        "horror",
        "mystery",
        "thriller",
        "historical",
        "adventure",
        "drama",
        "comedy",
        "crime",
        "biography",
        "psychology",
        "philosophy",
        "science",
        "economics",
        "politics"
    ]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    for doc, topic in zip(doc_names, topics):
        pseudo_text = create_pseudo_text(topic)
        result = classifier(pseudo_text, candidate_labels, multi_label=False)
        best_genre = result["labels"][:3]
        score = result["scores"][0]
        print(f"{doc} — likely genre: {best_genre} (confidence: {score:.2f})")

if __name__ == "__main__":
    main()
