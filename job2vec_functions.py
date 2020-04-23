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

def loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = tf.one_hot(y_true, depth=vocab_size)
    return -tf.tensordot(y_pred, tf.reduce_sum(y_true, axis=[1]),2)/batch_size + tf.reduce_sum(4*tf.math.log(tf.reduce_sum(tf.exp(y_pred), axis=[1])))/batch_size

class Job2vec(tf.keras.Model):
    def __init__(self, n_dim):
        super(Job2vec, self).__init__()
        self.W1 = tf.Variable(tf.random.uniform([vocab_size, n_dim],-1.0, 1.0))
        self.W2 = tf.Variable(tf.random.uniform([n_dim, vocab_size],-1.0, 1.0))
        
    def __call__(self, inputs, training=True):
        inputs = tf.one_hot(inputs, depth = vocab_size, axis=-1)
        inputs = tf.squeeze(inputs, axis=1)
        h = tf.linalg.matmul(inputs, self.W1)
        u = tf.linalg.matmul(h, self.W2)
        return u
    
def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1,index_word2,index_word3,vectors,number_closest):
    list1=[]
    query_vector = vectors[index_word1]-vectors[index_word2]+vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])
