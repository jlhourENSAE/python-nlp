### job2vec
### representation vectorielle des fiches metier ROME
### Idee de Christophe Gaillac
### Jeremy L'Hour
### 22/04/2020

### Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.enable_eager_execution()


### Loading data
#path = '/Users/jeremylhour/Documents/Python/job2vec/data/Textes sur les fiches ROME.csv'
path = '//ulysse/users/JL.HOUR/1A_These/A. Research/python-nlp/data/Textes sur les fiches ROME.csv'
with open(path,encoding='utf=8') as f:
    df = pd.read_csv(f)
    
df.shape

### Merge all text into one string for a given job

df['text'] = df[['ROME_PROFESSION_CARD_CODE','TXT_NAME']].groupby(['ROME_PROFESSION_CARD_CODE'])['TXT_NAME'].transform(lambda x: '.'.join(x))
new_df = df[['ROME_PROFESSION_CARD_CODE','text']].drop_duplicates()

raw_text = new_df['text'].str.cat()

### Filter stop words​
# liste des tokens
tokenizer = RegexpTokenizer('[a-zA-Zéèàù0-9+]{2,}')
word_list = tokenizer.tokenize(raw_text.lower())

# stemming
stemmer = FrenchStemmer()
word_list  = stemming(word_list)

# definition et suppression des stop words
stop_words = set(stopwords.words('french'))
stop_words.update(['secteur','peut','entrepris','activ','emploi','cet','professionnel',
                   'utilis','horair','traval','semain','impliqu','exig','précédent','spécialis',
                   'peuvent','selon','méti','sein','dan','différent'])
word_list = stop_words_filtering(word_list)

# fit vectorizer
vectorizer = CountVectorizer(token_pattern='[a-zA-Zéèàù0-9+]{2,}')
vectorizer.fit(word_list)

word2idx = vectorizer.vocabulary_
idx2word = dict(zip(word2idx.values(),word2idx.keys()))
vocab_size = len(word2idx)

### Mise en forme des donnees
WINDOW_SIZE = 5

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
        self.W1 = tf.Variable(tf.random.uniform([vocab_size, n_dim],-1.0, 1.0))
        self.W2 = tf.Variable(tf.random.uniform([n_dim, vocab_size],-1.0, 1.0))
        
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

job2vec = Job2vec(30)
optimizer = Adam(learning_rate = 0.01)
job2vec.compile(optimizer = optimizer,
                loss = loss)

training_history = job2vec.fit(X, y, epochs = 20, batch_size = 64)

optimizer = Adam(learning_rate = 0.001)
job2vec.compile(optimizer = optimizer,
                loss = loss)
training_history = job2vec.fit(X,y, epochs = 20, batch_size = 64)


### Post-process

vectors = job2vec.W1.numpy()
transformer = Normalizer()
vectors = transformer.fit_transform(vectors)
vectors.shape


### Define useful functions to evaluate model
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
        
        
### Reste à definir les vecteurs des codes ROME 
def get_job_vec(job_description):
    word_list = tokenizer.tokenize(job_description)
    word_list = stemming(word_list)
    filtered_words = [word for word in word_list if word not in stop_words]
    job = np.zeros(vectors.shape[1])
    for word in filtered_words:
        job = job + vectors[word2idx[word]]
    return job/len(filtered_words)
    
job1 = new_df['text'].iloc[0]
job2 = new_df['text'].iloc[1]
