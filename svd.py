from tdm_generator import build_matrix
import numpy as np
import sys

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

if __name__ == "__main__":
    main()
