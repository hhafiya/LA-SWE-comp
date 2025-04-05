from tdm_generator import build_matrix
from svd import svd
import numpy as np
import matplotlib.pyplot as plt
import sys

def pca(V, k):
    V_centered = V - np.mean(V, axis = 0)

    U, S, Vt = svd(V_centered)

    p_components = np.transpose(Vt)[:k]
    V_pca = V_centered @ p_components.T

    res_variance = (S ** 2) / (V.shape[0]-1)
    res_variance = res_variance[:k]

    return V_pca, p_components, res_variance



def main():
    if len(sys.argv) < 1:
        print("Usage: python3 pca.py")
        return
    
    input_folder = "data/documents"
    stop_words = "data/stop_words/stopwords.txt"
    name_path = "data/stop_words/names.txt"

    doc_names, vocabulary, term_doc_matrix = build_matrix(input_folder, stop_words, name_path)

    V_pca, p_components, res_variance = pca(term_doc_matrix, 2)


    

if __name__ == "__main__":
    main()

