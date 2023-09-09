import pandas as pd
import re  # For preprocessing
import pandas as pd  # For data handling
import numpy as np
from langdetect import detect
import swifter 
from symspellpy import SymSpell, Verbosity
import pkg_resources
import argparse

parser = argparse.ArgumentParser(description='Data cleaning script for text data')

parser.add_argument('--path', type=str, default='/Users/shafz/OneDrive/Documents/deep-learning-final-project-yelp_reviews_classification/data/raw/',
                     metavar='P', help='path to folder with files that needs to be cleaned')


################### Global variables ###################################
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

############ Functions ###########################################
# function to find the language in which reviews were written 
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang

# function for spell checking 
def spell_check_text(text):
    words = text.split()
    checked_words = []
    for word in words:
        # Check if word is in dictionary
        if sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=0):
            checked_words.append(word)
        else:
            # Get suggestions for misspelled word
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if len(suggestions) > 0:
                checked_words.append(suggestions[0].term)
            else:
                checked_words.append(' ')
    return ' '.join(checked_words)

# function for handling contractions
def expand_contractions(text):
  contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}
  
  contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

  def replace(match):
    return contractions_dict[match.group(0)]
  
  return contractions_re.sub(replace, text)


###### Data Cleaning #############################

def clean_data(df):
    # finding reviews in english 
    df['language'] = df['text'].swifter.apply(detect_language)
    restaurants_english = df[restaurants_filtered['language'] == 'en']

    # expanding contractions
    restaurants_english['cleaned'] = restaurants_english['text'].apply(lambda x:expand_contractions(x))

    ## Making all words lowercase, removing punctuation, URLs, and white spaces 
    restaurants_english['cleaned'] = restaurants_english['cleaned'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', ' ', x))
    restaurants_english['cleaned'] = restaurants_english['cleaned'].str.lower().apply(lambda x: re.sub(r"[\d\n\-\./]+", ' ', x))
    restaurants_english['cleaned'] = restaurants_english['cleaned'].apply(lambda x: re.sub(' +',' ',x))
    restaurants_english['cleaned'] = restaurants_english['cleaned'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # fixing any bugs in encoding 
    restaurants_english['cleaned'] = restaurants_english['cleaned'].str.normalize('NFKD')\
        .str.encode('ascii', errors='ignore')\
        .str.decode('utf-8')


    # finding any empty strings and setting to nans
    restaurants_english['cleaned'] = restaurants_english['cleaned'].replace(' ', np.nan)
    restaurants_english.dropna(inplace=True)

    restaurants_english.drop('language', axis = 1, inplace=True)

    ####### Spelling Checking ##############

    custom_words = [('waitlist', 100)]
    for word, freq in custom_words:
        sym_spell.create_dictionary_entry(word, freq)

    restaurants_english['spell_checked'] = restaurants_english['cleaned'].swifter.apply(spell_check_text)
    restaurants_english['spell_checked'] = restaurants_english['spell_checked'].apply(lambda x: re.sub(' +',' ',x))


    restaurants_english.to_csv('./reviews_cleaned.csv', index=False)
    print(' ----------------------------- Cleaned data saved ! --------------------------')

if __name__ == '__main__':

    ## Load data and filter out restaurants 
    args = parser.parse_args()
    path = args.path 

    business = pd.read_csv(path+'raw_business.csv')
    restaurants_ids = business[business.categories.fillna('-').str.lower().str.contains('restaurant')]

    reviews = pd.read_csv(path+'raw_reviews.csv')
    reviews.drop('Unnamed: 0', axis =1, inplace = True)
    restaurants = reviews[reviews['business_id'].isin(restaurants_ids['business_id'].tolist())]

    restaurants_filtered = restaurants.groupby('business_id').filter(lambda x : len(x)>1000)
    restaurants_filtered = restaurants_filtered[['business_id', 'stars', 'text']]

    clean_data(restaurants_filtered) 
