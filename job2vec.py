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

### Quick clean-up
clean_val = {
        ',': '',
        'é': 'e',
        'è': 'e',
        'à': 'a',
        'â': 'a',
        '\(': '',
        '\)': '',
        '\.\.\.': '',
        "''": "'",
             }
df.replace({'TXT_NAME' : clean_val},inplace=True,regex=True)

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
                   'utilis','horair','traval','semain','impliqu','exig','précédent','specialis',
                   'peuvent','selon','meti','sein','dan','different','travail','acces'])
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
batch_size = 64

job2vec = Job2vec(100)
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

### Reste à definir les vecteurs des codes ROME 
def get_job_vec(job_description):
    word_list = tokenizer.tokenize(job_description)
    word_list = stemming(word_list)
    filtered_words = [word for word in word_list if word not in stop_words]
    job = np.zeros(vectors.shape[1])
    for word in filtered_words:
        try : job = job + vectors[word2idx[word]]
        except: continue
    return job/len(filtered_words)
    
job_vectors = new_df['text'].apply(func = get_job_vec)

job1 = new_df['text'].iloc[0]
job2 = new_df['text'].iloc[1]

index_closest_words = find_closest(1, job_vectors, 10)

cosine_similarity(get_job_vec(job1),get_job_vec(job2))
