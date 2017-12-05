# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:39:41 2017

@author: xILENCE
"""

import nltk
from nltk.tokenize import word_tokenize #seperate the phrase into array of word
from nltk.stem import WordNetLemmatizer #run ran etc.. will be the same word, same for -ing -ed etc...
import numpy as np
import random #will help shuffle the data 
import pickle #will save data
from collections import Counter #count stuffs


lemmatizer = WordNetLemmatizer()
hm_lines = 10000000



def create_lexicon(pos,neg):
    
    lexicon = []
    
    for fi in [pos, neg]:   #iterrate for each lines in pos, neg file
        
        with open(fi,'r',encoding='cp437') as f:  #read each lines of the file
            contents = f.readlines()
            for l in contents[:hm_lines]:   #for loop up to how many lines
                all_words = word_tokenize(l.lower())    #lower put all words in lower case, tokenize is seperating the words into array
                lexicon += list(all_words)   #add each word from the all_words array into lexicon
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]    #lemmatize (aka. replace similar word by the same word/conjugaison)
    w_counts = Counter(lexicon) #apply counter to lexicon, the output is something like : {'the': 54642,'and': 25642}
    
    l2 = []
    
    for w in w_counts:
        if 1000 > w_counts[w] > 50:     #we don't want super common words because it's not needed to see if a phrase is pos or neg
            l2.append(w)    #give w into l2, so we take only the word that occur between 1000 et 50, the rest is not taken
    print(len(l2))        
    return l2


def sample_handling(sample, lexicon, classification):
    featureset=[]
    
    with open(sample, 'r',encoding='cp437') as f: #same as before, this open the file, manipulate it and close it afterward
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon)) #create array of zeros size of lexicon
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())   #index value of the word.lower
                    features[index_value] += 1  
                features = list(features)
                featureset.append([features, classification]) #give the array of features and classification into featureset
    
            
        '''
        featureset will look like :
            [
            [[0 1 1 0 0 0 ...], [0 1]],
            [[0 0 0 1 0 0 ...], [1 0]],
            [],
            ...
            ]
            the first is the words recognized from the dataset in the new phrase and the second is whether or not the phrase is neg or pos [0 1] or [1 0]
        '''
    
    return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):  #test_size 0.1 = 10% test_size
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('dataset_sentiments/pos.txt',lexicon,[1,0])     #classificaton for positiv is 1 0  
    features += sample_handling('dataset_sentiments/neg.txt',lexicon,[0,1])
    random.shuffle(features)    #we shuffle to make the data more real and train the network in a real way
    
    features = np.array(features) #convert to np array
    
    testing_size = int(test_size*len(features))     #we will train 10% of features
    
    train_x = list(features[:,0][:-testing_size])   #we take all rows of 1st colomn
    train_y = list(features[:,1][:-testing_size])   #beginning 'till -testing size (we train on 10% and test on the rest)
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])    #-testing size 'till end
    
    
    return train_x,train_y,test_x,test_y
    
    
if __name__ == '__main__':     #allow the execution of this loop only if execute directly and not if import into a file
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('dataset_sentiments/pos.txt','dataset_sentiments/neg.txt')
    with open('dataset_sentiments/sentiment_set.pickle','wb') as f:   #open as writing binary
        pickle.dump([train_x,train_y,test_x,test_y], f) #dump thoses values because we don't need them after 1 execution of the script to pass them into the network
        
        
        
    
    
    
    
    
    
    
    
    











