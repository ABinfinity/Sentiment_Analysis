import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk

#!/usr/bin/env python

import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences


if __name__=='__main__':
	train=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','labeledTrainData.tsv'), \
		header=0, delimiter="\t", quoting=3)

	test=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','testData.tsv'), header=0, \
		delimiter="\t", quoting=3)

	print ("The first review is...")

	print (train["review"][0])
	raw_input("press enter to continue...")

	clean_train_reviews=[]

	print("Cleaning and parsing the training set movie reviews...\n")

	for i in xrange(0, len(train["reviews"])):
		clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i],True)))


	print ("Creating the bag of words...\n")

	vectorizer = CountVectorizer(analyse="word" , tokeniser= None , preprocessor= None , \
		stop_words= None , max_features= 5000) 


	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()

	print("Training the random forest(This may take a while)....")
	forest = RandomForestClassifier(n_estimators = 100)

	forest = forest.fit( train_data_features , train["sentiments"])

	clean_test_reviews = []

	print ("Cleaning and parsing the test data set movie reviews...\n")
	for i in xrange(0, test["reviews"]):
		clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))
	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()


	print("Predicting test labels...\n")
	result = forest.predict(test_data_features)
	output = pd.DataFrame( data = { "id":test["id"] , "sentiments":result})
	output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'sentiment_analysis.csv') , index = False ,  quoting = 3)
	print ("Wrote results to sentiment_analysis.csv")

