### job2vec
### representation vectorielle des fiches metier ROME
### Idee de Christophe Gaillac
### Jeremy L'Hour
### 22/04/2020

### Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import tensorflow as tf


### Loading data
with open('/Users/jeremylhour/Documents/Python/job2vec/data/Textes sur les fiches ROME.csv') as f:
    df = pd.read_csv(f)
    
df[df.ROME_PROFESSION_CARD_CODE=='A1101'].shape

### Merge all text into one string for a given job

df['text'] = df[['ROME_PROFESSION_CARD_CODE','TXT_NAME']].groupby(['ROME_PROFESSION_CARD_CODE'])['TXT_NAME'].transform(lambda x: '.'.join(x))
new_df = df[['ROME_PROFESSION_CARD_CODE','text']].drop_duplicates()

raw_text = new_df['text'].str.cat()

### Filter stop words​
# liste des tokens
tokenizer = RegexpTokenizer('[a-zA-Zéèàù]{2,}')
word_list = tokenizer.tokenize(raw_text.lower())

# stemming
stemmer = FrenchStemmer()
​​
def stemming(mots):
    racines = []
    for mot in mots:
        radical = stemmer.stem(mot)
        racines.append(radical)
    racines = set(racines)
    return list(racines)
​
word_list = stemming(word_list)
​
# definition et suppression des stop words
stop_words = stopwords.words('french')
​
def stop_words_filtering(x):
    for word in stop_words:
        try: x.remove(word)
        except: continue
    return x
​
word_list =  stop_words_filtering(word_list)
​
# fit vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(word_list)

word2idx = vectorizer.vocabulary_
idx2word = dict(zip(word2idx.values(),word2idx.keys()))
​
vocab_size = len(word2idx)

### Mise en forme des donnees
def sentenceToData(tokens,WINDOW_SIZE):
    window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
    X,Y=([],[])
    for word_index, word in enumerate(tokens) :
        if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
            X.append(word2idx[word])
            Y.append([word2idx[tokens[word_index-i]] for i in window])
    return X, Y
​
WINDOW_SIZE = 5
​
X, Y = ([], [])
for sentence in raw_text.lower().split("."):
    word_list = tokenizer.tokenize(sentence)
    word_list = stemming(word_list)
    filtered_words = [word for word in word_list if word not in stop_words]
    X1, Y1 = sentenceToData(filtered_words, WINDOW_SIZE//2)
    X.extend(X1)
    Y.extend(Y1)
    
X = np.array(X).astype(int).reshape([-1,1])
y = np.array(Y).astype(int)
print('Shape of X :', X.shape)
print('Shape of Y :', y.shape)

### Preparation du modele word2vec
class Job2vec(tf.keras.Model):
    def __init__(self, n_dim):
        super(Job2vec, self).__init__()
        self.W1 = tf.Variable(tf.random.normal([vocab_size, n_dim]))
        self.W2 = tf.Variable(tf.random.normal([n_dim, vocab_size]))
        
    def __call__(self, inputs, training=True):
        inputs = tf.one_hot(inputs, depth = vocab_size, axis=-1)
        inputs = tf.squeeze(inputs, axis=1)
        h = tf.linalg.matmul(inputs, self.W1)
        u = tf.linalg.matmul(h, self.W2)
        return u
    
batch_size = 64
def loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = tf.one_hot(y_true, depth=vocab_size)
    return -tf.tensordot(y_pred, tf.reduce_sum(y_true, axis=[1]),2)/batch_size + tf.reduce_sum(4*tf.math.log(tf.reduce_sum(tf.exp(y_pred), axis=[1])))/batch_size

from tensorflow.keras.optimizers import Adam
​
job2vec = Jord2vec(30)
optimizer = Adam(learning_rate = 0.01)
