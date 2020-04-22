# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:58:42 2020
Job2vec useful functionss
@author: jl.hour
"""
def stemming(mots):
    racines = []
    for mot in mots:
        radical = stemmer.stem(mot)
        racines.append(radical)
    racines = set(racines)
    return list(racines)

def stop_words_filtering(x):
    for word in stop_words:
        try: x.remove(word)
        except: continue
    return x

def sentenceToData(tokens,WINDOW_SIZE):
    window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
    X,Y=([],[])
    for word_index, word in enumerate(tokens) :
        if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
            X.append(word2idx[word])
            Y.append([word2idx[tokens[word_index-i]] for i in window])
    return X, Y
