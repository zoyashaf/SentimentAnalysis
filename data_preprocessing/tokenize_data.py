import spacy  # For preprocessing
import pandas as pd
import re  # For preprocessing
import pandas as pd  # For data handling
import numpy as np
import swifter 
import argparse

parser = argparse.ArgumentParser(description='Data tokenization for text data')

parser.add_argument('--path', type=str, default='/Users/shafz/OneDrive/Documents/deep-learning-final-project-yelp_reviews_classification/data/raw/',
                     metavar='P', help='path to folder with cleaned file')


if __name__ == '__main__':

  # Load data and filter out restaurants 
  args = parser.parse_args()
  path = args.path 

  restaurants = pd.read_csv(path + '/reviews_cleaned.csv')

  nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
  nlp.Defaults.stop_words -= {'few', 'whereas', 'unless', 'anything', 'everything', 'nothing','but', 'before', 
                              'afterwards', 'besides',  'anywhere', 'everyone', 'however', 'against', 'never', 
                              'cannot', 'even', 'neither', 'empty', 'nor','well', 'either', 'least', 'less', 'none', 
                              'side', 'not', 'full'}
  nlp.Defaults.stop_words |= {'food', 'service', 'drinks', 'restaurant', 'come', 'place', 'get', 'go'}

  texts = restaurants['cleaned'].tolist()
  lemmatized_texts = []
  for doc in nlp.pipe(texts, batch_size=1000, n_process=4):
      lemmatized_texts.append(' '.join([token.lemma_ for token in doc if (token.is_stop==False)]))

  restaurants['lemmatized'] = lemmatized_texts

  restaurants.to_csv('./lemmatized_reviews.csv')