import numpy as np # linear algebra
import pandas as pd # data processing

# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score


# Libraries and packages for text (pre-)processing 
import string
import re
import nltk 
# nltk.download('stopwords')

from gensim.models import Word2Vec

# For type hinting
from typing import List



class Tokenizer: 
    """ After cleaning and denoising steps, in this class the text is broken up into tokens.
    if clean: clean the text from all non-alphanumeric characters,
    if lower: convert the text into lowercase,
    If de_noise: remove HTML and URL components,
    if remove_stop_words: and remove stop-words,
    If keep_neagation: attach the negation tokens to the next token 
     and treat them as a single word before removing the stopwords
     
    Returns:
    List of tokens
    """
    # initialization method to create the default instance constructor for the class
    def __init__(self,
                 clean: bool = True,
                 lower: bool = True, 
                 de_noise: bool = True, 
                 remove_stop_words: bool = True,
                keep_negation: bool = True) -> List[str]:
      
        self.de_noise = de_noise
        self.remove_stop_words = remove_stop_words
        self.clean = clean
        self.lower = lower
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.keep_negation = keep_negation

    # other methods  
    def denoise(self, text: str) -> str:
        """
        Removing html and URL components
        """
        html_pattern = r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"

        text = re.sub(html_pattern, " ", text)
        text = re.sub(url_pattern," ",text).strip()
        return text
       
    
    def remove_stopwords(self, tokenized_text: List[str]) -> List[str]:
        text = [word for word in tokenized_text if word not in self.stopwords]
        return text

    
    def keep_negation_sw(self, text: str) -> str:
        """
        A function to save negation words (n't, not, no) from removing as stopwords
        """
        # to replace "n't" with "not"
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        # to join not/no into the next word
        text = re.sub("not ", " NOT", text)
        text = re.sub("no ", " NO", text)
        return text
    
    
    def tokenize(self, text: str) -> List[str]:
        """
        A function to tokenize words of the text
        """
        non_alphanumeric_pattern =r"[^a-zA-Z0-9]"
        
        # to substitute multiple whitespace with single whitespace
        text = str(text)  
        text = ' '.join(text.split())

        if self.de_noise:
            text = self.denoise(text)
        if self.lower:
            text = text.lower()
        if self.keep_negation:
            text = self.keep_negation_sw(text)
            
        if self.clean:
            # to remove non-alphanumeric characters
            text = re.sub(non_alphanumeric_pattern," ", text).strip()

        tokenized_text = text.split()

        if self.remove_stop_words:
            tokenized_text = self.remove_stopwords(tokenized_text)

        return tokenized_text

### CODE ABOVE IS TO PUT THE TOKENIZE INTO A VECTOR AND CHECK SOMETHING WAS OVERLOOK
### WHEN DATA WAS CLEANING

def w2v_trainer(doc_tokens: List[str],
                epochs: int = 10,
                workers: int = 3,
                vector_size: int = 300,
                window: int = 5,
                min_count: int = 2):
    """ 
    Going through a list of lists, where each list within the main list contains a set of tokens from a doc, this function trains a Word2Vec model,
    then creates two objects to store keyed vectors and keyed vocabs   
    Parameters:
    doc_tokens   : A tokenized document 
    epochs       : Number of epochs training over the corpus
    workers      : Number of processors (parallelization)
    vector_size  : Dimensionality of word embeddings
    window       : Context window for words during training
    min_count    : Ignore words that appear less than this
    Returns:
    keyed_vectors       : A word2vec vocabulary model
    keyed_vocab 
    
    """
    w2v_model = Word2Vec(doc_tokens,
                         epochs=10,
                         workers=3,
                         vector_size=300,
                         window=5,
                         min_count=2)
    
    # create objects to store keyed vectors and keyed vocabs
    keyed_vectors = w2v_model.wv
    keyed_vocab = keyed_vectors.key_to_index
    
    return keyed_vectors, keyed_vocab
    
    
""" 
DOES NOT WORK:
def calculate_ossa_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    polarity_scores = []
    subjectivity_scores = []
    for key, value in scores.items():
        if key == "compound":
            continue
        if value > 0:
            polarity_scores.append(value)
        else:
            subjectivity_scores.append(abs(value))
    if len(polarity_scores) == 0 or len(subjectivity_scores) == 0:
        return 0
    ossa_sentiment = sum(polarity_scores) / len(polarity_scores) - sum(subjectivity_scores) / len(subjectivity_scores)
    return ossa_sentiment
    
"""
def calculate_overall_similarity_score(keyed_vectors,
                             target_tokens: List[str], 
                             doc_tokens: List[str]) -> float:
    """
    Going through a tokenized doc, this function computes vector similarity between 
    doc_tokens and target_tokens as 2 lists by n_similarity(list1, list2) method based on 
    Word2Vec vocabulary (keyed_vectors), 
    then returns the similarity scores.  
    
    Parameters:
    target_tokens  : A set of semantically co-related words  
    doc_tokens     : A tokenized document 
    keyed_vectors  : A word2vec vocabulary model
    
    Returns:
    vector similarity scores between 6 tokenized list doc_tokens and target_tokens  
    """
    
    target_tokens = [token for token in target_tokens if token in keyed_vectors]

    doc_tokens = [token for token in doc_tokens if token in keyed_vectors]
    
    if not (target_tokens and doc_tokens):
        return 0.0
    else:
        similarity_score = keyed_vectors.n_similarity(target_tokens, doc_tokens)
        return similarity_score


def overall_semantic_sentiment_analysis (keyed_vectors, 
                                         care_target_tokens: List[str],
                                         fair_target_tokens: List[str],
                                         loyal_target_tokens: List[str],
                                         auth_target_tokens: List[str],
                                         san_target_tokens: List[str],
                                         lib_target_tokens: List[str],
                                         doc_tokens: List[str], 
                                         doc_is_series: bool = True) -> float:
    """   
    Parameters:
    keyed_vectors           : A word2vec vocabulary model
    care_target_tokens      : A list of sentiment or opinion words that indicate care/harm opinions 
    fair_target_tokens      : A list of sentiment or opinion words that indicate fairness/cheating opinions 
    loyal_target_tokens     : A list of sentiment or opinion words that indicate loyalty/betrayal opinions 
    auth_target_tokens      : A list of sentiment or opinion words that indicate authority/subversion opinions 
    san_target_tokens       : A list of sentiment or opinion words that indicate sanctity/degradation opinions 
    lib_target_tokens       : A list of sentiment or opinion words that indicate liberty/oppression opinions 
    doc_tokens              : A tokenized document 
    
    
    Returns:
    care_score : vector similarity scores between doc_tokens and care_target_tokens
    fair_score : vector similarity scores between doc_tokens and fair_target_tokens
    loyal_score : vector similarity scores between doc_tokens and loyal_target_tokens
    auth_score : vector similarity scores between doc_tokens and auth_target_tokens
    san_score : vector similarity scores between doc_tokens and san_target_tokens
    lib_score : vector similarity scores between doc_tokens and lib_target_tokens
    
    semantic_sentiment_score  : maximum score between the scores
    semantic_sentiment_polarity : Overall score: 0 for more care, 1 for more fair , 2 for more loyal, 3 for more auth, 4 for san, 5 for more lib
    """
## # Check that the input lists are not empty
#    if not (positive_target_tokens and negative_target_tokens and doc_tokens):
#        raise ValueError("At least one of the passed lists is empty.")
    
    care_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=care_target_tokens, 
                                                                 doc_tokens=x))
    
    fair_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=fair_target_tokens, 
                                                                 doc_tokens=x))
    
    loyal_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=loyal_target_tokens, 
                                                                 doc_tokens=x))
    
    auth_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=auth_target_tokens, 
                                                                 doc_tokens=x))
    
    san_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=san_target_tokens, 
                                                                 doc_tokens=x))
    
    lib_score = doc_tokens.apply(lambda x: calculate_overall_similarity_score(keyed_vectors=keyed_vectors, 
                                                                 target_tokens=lib_target_tokens, 
                                                                 doc_tokens=x))
    
    # Find the maximum score among care_score, fair_score, loyal_score, auth_score, san_score, lib_score
    max_score = pd.concat([care_score, fair_score, loyal_score, auth_score, san_score, lib_score], axis=1).max(axis=1)
    
# Determine the semantic sentiment polarity based on the maximum score
    semantic_sentiment_polarity = []

    for index, value in max_score.items():
        if value == care_score[index]:
            semantic_sentiment_polarity.append(0)
        elif value == fair_score[index]:
            semantic_sentiment_polarity.append(1)
        elif value == loyal_score[index]:
            semantic_sentiment_polarity.append(2)
        elif value == auth_score[index]:
            semantic_sentiment_polarity.append(3)
        elif value == san_score[index]:
            semantic_sentiment_polarity.append(4)
        elif value == lib_score[index]:
            semantic_sentiment_polarity.append(5)
            
    s = pd.DataFrame(semantic_sentiment_polarity)

              
    return care_score, fair_score, loyal_score, auth_score, san_score, lib_score, max_score, s     
    
def evaluate_model (y_true: pd.Series, 
                              y_pred: pd.Series, 
                              report:bool = False,
                              plot: bool = False)-> float:
    
    # A function to calculate F1, Accuracy, Recall, and Precision Score
    # If report: it prints classification_report 
    # If plot: it prints Confusion Matrix Heatmap
    
    if report:
        print(classification_report(y_true, 
                            y_pred,
                            digits=4))
    if plot:
        # figure
        fig, ax = plt.subplots(figsize=(4, 4))
        conf_matrix = pd.crosstab(y_true, 
                           y_pred, 
                           rownames=['Actual'], 
                           colnames=['Predicted'])
        sns.heatmap(conf_matrix, 
                    annot=True, fmt=".0f",
                    cmap='RdYlGn', # use orange/red colour map
                    cbar_kws={'fraction' : 0.04}, # shrink colour bar
                    linewidth=0.3, # space between cells
                   ) 
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()
        
    if not report and not plot:
        print('* Accuracy Score: ', "{:.4%}".format(accuracy_score(y_true, y_pred)))
        print('* F1 Score: ', "{:.4%}".format(f1_score(y_true, y_pred, average='micro' )))
        print('* Recall Score: ', "{:.4%}".format(recall_score(y_true , y_pred, average='micro'  )))
        print('* Precision Score: ', "{:.4%}".format(precision_score(y_true , y_pred, average='micro' )))
         