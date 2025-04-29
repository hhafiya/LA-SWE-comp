# LA-project Text Genre Classifier
### Authors
This project was made with love by: \
    https://github.com/hhafiya \
    https://github.com/shshrg \
    https://github.com/linyvez 

### Goal and Idea
The goal of this project is to develop a simple machine learning model for the automatic classification of book genres based on their texts. We want to streamline the process of uploading books into a digital system, where each book must be assigned to an appropriate genre category.

Matrix factorization (specifically, SVD) is used to generate static word embeddings. 
These embeddings are then fed into a text classifier based on a softmax model to predict
the genre of a given text.

### Usage
First, install the necessary requirements from ```requirements.txt```. Then, the program can be used in two ways:
1. Using ```train.py``` and ```predict.py``` scripts (*recommended method*)
   
   - The ```train.py``` script is used to generate necessary data from the training dataset.

   - **Important**: We have already provided the necessary data in ```data/train_results```. Unless you want to train the model on your own data, there is no need to rerun this script and you can go straight to the prediction step.
   
   - The ```predict.py``` script is used to predict the genre(s) of books. You can run it via:
     ```
      python predict.py path_to_folder_with_books
     ```
     *Note*: Only ```.txt``` files are currently supported, so please make sure all the books in the folder are in that format.
2. Using the Jupyter Notebook

   This notebook was mainly made for testing and tweaking the model, but you can still use it for testing predictions on some data. Make sure all the paths and the labels of documents are correct, and then run the cells. 
