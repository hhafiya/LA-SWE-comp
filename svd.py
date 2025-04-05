from tdm_generator import build_matrix
import numpy as np
import sys
from transformers import pipeline
import logging

# logging.getLogger("transformers").setLevel(logging.ERROR)

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
            v_i = np.zeros(matrix.shape[0])
        right_sing_matrix_V.append(v_i)
    right_sing_matrix_V = np.column_stack(right_sing_matrix_V)

    return left_sing_matrix_U, sigma_E, right_sing_matrix_V

def main():
    if len(sys.argv) < 1:
        print("Usage: python3 svd.py")
        return
    
    input_folder = "data/documents"
    stop_words = "data/stop_words/stopwords.txt"
    name_path = "data/stop_words/names.txt"

    doc_names, vocabulary, term_doc_matrix = build_matrix(input_folder, stop_words, name_path)

    U, E, V = svd(term_doc_matrix)

    classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli")
    
    dimension_labels = []

    candidate_labels = [
        "fantasy",
        "romance",
        "science fiction",
        "horror",
        "thriller",
        "adventure",
        "drama",
        "comedy",
        "crime",
        "psychology",
        "philosophy",
        "science",
        "economics",
        "politics"
    ]

    for d in range(V.shape[1]): 
        dimension_weights = V[:, d]

        top_indices = np.argsort(np.abs(dimension_weights))[::-1]
        top_words = [vocabulary[i] for i in top_indices[:30]]
        dimension_description = " ".join(top_words)

        result = classifier(dimension_description, candidate_labels)

        top3_indices = sorted(range(len(result["scores"])), key=lambda i: result["scores"][i], reverse=True)[:3]
        top3_genres = [result["labels"][i] for i in top3_indices]
        dimension_labels.append(top3_genres)
        
    doc_topic_vectors = U @ E

    doc_to_topic = []

    for i in range(V.shape[1]):
        row_vec = doc_topic_vectors[i, :]
        best_dim = np.argmax(np.abs(row_vec))
        doc_topic = dimension_labels[best_dim]
        doc_to_topic.append(doc_topic)
    
    for i, topic in enumerate(doc_to_topic):
        print(f"{doc_names[i].strip('.txt')} => {topic}")

if __name__ == "__main__":
    main()
