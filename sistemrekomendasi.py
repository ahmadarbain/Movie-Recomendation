# -*- coding: utf-8 -*-
"""SistemRekomendasi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JbyKPreM7P_CoipiJAgOqzEOzQmrfi0p

# **Data Preparation**

## Load Data

### Menghubungkan ke Google Drive
"""

from google.colab import drive

drive.mount('/content/drive')

"""## Download Dataset Kaggle

### Memasang(*Install*) Kaggle
"""

! pip install -q kaggle

"""### Change Directory"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Dataset/Kaggle

"""### Membuat Folder Film Rekomendasi"""

!mkdir '/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation'

"""### Masuk ke folder kerja Film Recomendation"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation

"""### Mengunggah API Kaggle ke folder kerja"""

from google.colab import files

files.upload()

"""### Konfigurasi ke path "Kaggle.json""""

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation"

"""### Download Dataset dari Kaggle"""

!kaggle datasets download -d rounakbanik/the-movies-dataset

"""### Unzip Dataset"""

!unzip '/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation/the-movies-dataset.zip' -d '/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation'

"""# **Univariate Exploratory Data Analysis**

## Import Library dan Read Dataset CSV
"""

import numpy as np
import pandas as pd
import nltk
import re
re.compile('<title>(.*)</title>')
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from zipfile import ZipFile
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

data1 = pd.read_csv('/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation/ratings_small.csv')
data2 = pd.read_csv('/content/drive/MyDrive/Dataset/Kaggle/Film Rekomendation/movies_metadata.csv')

"""## Melihat Dataset

### Melihat dataset "data1" bagian head
"""

data1.head()

"""### Melihat dataset "data2" bagian head"""

data2.head()

"""## Melakukan konfigurasi pada kolom genres

Mengambil bagian genres nya saja yaitu menghapus variabel id berserta nilainya dan juga menghapus variabel nama
"""

data2['genres'] = data2['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x]if isinstance(x,list) else [])

"""## Melihat dataset "data2" setelah dilakukan perubahan"""

data2.head()

"""## Melihat Data Info

### Melihat data info pada data1
"""

data1.info()

"""### Melihat data info pada data3"""

data2.info()

"""## Melihat Jumalah Data Baris dan Kolom"""

print("Jumlah dataset pada data1 : ",data1.shape)
print("============================")
print("Jumlah Dataset pada data2 : ",data2.shape)

"""## Mengecek Missing Value Dataset """

print("Dataset 1")
print(data1.isnull().sum())
print("======================================")
print("Dataset 2")
print(data2.isnull().sum())

"""# **Data Preprocessing**

## Mengubah data list menjadi string
"""

data2["gen_list"] = data2["genres"]
data2["gen_list"] = data2["gen_list"].apply(lambda x: " ".join(x))

"""## Mengecek data2"""

data2.head()

"""## Menambahkan kolom genres ke dalam kolom overview"""

data2.overview = data2.gen_list + " " + data2.overview
data2.head()

"""## Menghapus nilai null pada kolom overview"""

data2["overview"].dropna(inplace = True)
data2["overview"]

"""# **Data Preparation**

## Cleaning Data Overview

### Menghapus tag HTML dan Angka
"""

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

cleaning = "@\S+|https?:\S+|http?:\S|[^A-Za-z]+"

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(cleaning, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

"""### Menerapkan hasil cleaning ke data overview"""

data2.overview = data2.overview.apply(lambda x: preprocess(x, stem = False))
data2.overview

"""# **Model Development : Collaborative Filtering**

## **Data Understanding**

### Membuat Dataset dengan Data Kurang Dari 2000
"""

data1 = data1[data1["userId"] < 2000]

"""### Melihat Dataset """

data1

"""### Menghapus kolom timestamp"""

data1 = data1.drop(['timestamp'], axis=1)
data1

"""## **Data Preprocessing**

### Menyimpan dataset data1 pada variabel
"""

rating_movie = data1
rating_movie.head()

"""### Membuat data baru dari dataset 2

* Membuat dataset baru berisi kolom title dan genre
* Melakukan cek nilai teratas
"""

movie_names = data2
movie_names = movie_names[['title', 'genres']]
movie_names.head()

"""### Menggabungkan dataset dari data1 dan data2

* data1 = data yang sudah dibersihkan dan di simpan pada variabel rating_movie
* data2 = dataset baru yang diambil dari data title dan genre yang disimpan dalam variabel movie_names
"""

movie_data = pd.concat([rating_movie, movie_names], axis=1)
movie_data.head()

"""## Data preparation 

* Mengurutkan nilai dari movieId
"""

preparation = movie_data
preparation.sort_values('movieId')

"""### Menghapus Nilai Duplikat movieId"""

preparation = preparation.drop_duplicates('movieId')
preparation

"""### Menghapus Data Kosong (NaN)"""

data_preparation = preparation.dropna()

"""### Mengkonversi Data Menjadi List"""

movie_id = data_preparation['movieId'].tolist() 
movie_name = data_preparation['title'].tolist()
movie_genres = data_preparation['genres'].tolist()

print(len(movie_id))
print(len(movie_name))
print(len(movie_genres))

"""### Membuat Directionary Data """

new_movie = pd.DataFrame({
    'id': movie_id,
    'title': movie_name,
    'genres': movie_genres,
})
new_movie

"""### Melihat Missing Value"""

new_movie.isnull().sum()

"""### Melakukan Encoding userId

---

1.   Merubah data userId menjadi List dan Memiliki nilai yang unik (Tidak sama)
2.   Melakukan Encoding Data userId
3.   Melakukan encoding angka ke userId 

---




"""

user_ids = data1['userId'].unique().tolist()
print('list userId: ', user_ids)
 
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userId : ', user_to_user_encoded)
 
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userId: ', user_encoded_to_user)

"""### Melakukan Encoding movieId

---

1.   Merubah data movieId menjadi List dan Memiliki nilai yang unik (Tidak sama)
2.   Melakukan Encoding Data movieId
3.   Melakukan encoding angka ke movieId 

---
"""

movie_ids = data1['movieId'].unique().tolist()
print('list movieId: ', movie_ids) 

movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
print('encoded movieId : ', movie_to_movie_encoded) 

movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
print('encoded angka ke movieId: ', movie_encoded_to_movie)

"""### Mapping Data

---
1.   Mapping userID ke dataframe user
2.   Mapping movieId ke dataframe movie

---



"""

data1['user'] = data1['userId'].map(user_to_user_encoded)
data1['movie'] = data1['movieId'].map(movie_to_movie_encoded)

"""### Melakukan Verifikasi Data

---
1.   Mendapatkan jumlah user
2.   Mendapatkan jumlah movie
3.   Mengubah rating menjadi nilai float
4.   Nilai minimum rating
5.   Nilai maximum rating
---





"""

num_users = len(user_to_user_encoded)
print(num_users)

num_movie = len(movie_encoded_to_movie)
print(num_movie)
 
data1['rating'] = data1['rating'].values.astype(np.float32)
min_rating = min(data1['rating'])
max_rating = max(data1['rating'])
 
print('Number of User: {}, Number of Movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))

"""### Rendom Distribusi Dataset"""

df = data1.sample(frac=1, random_state=42)
df

"""## Split Data Manjadi Training 80% dan Validasi 20%


---
1.   Membuat variabel x untuk mencocokkan data user dan resto menjadi satu value
2.   Membuat variabel y untuk membuat rating dari hasil 
3.   Membagi menjadi 80% data train dan 20% data validasi


---





"""

x = df[['user', 'movie']].values
 
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
 
print(x, y)

"""## **Training**

### Membuat Fungsi Rekomendasi
"""

class RecommenderNet(tf.keras.Model):
 
  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( 
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) 
    self.resto_embedding = layers.Embedding( 
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_movie, 1) 
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) 
    user_bias = self.user_bias(inputs[:, 0])
    resto_vector = self.resto_embedding(inputs[:, 1]) 
    resto_bias = self.resto_bias(inputs[:, 1]) 
 
    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2) 
 
    x = dot_user_resto + user_bias + resto_bias
    
    return tf.nn.sigmoid(x)

"""### Compile model Dengan Metode BinaryCrossentropy"""

model = RecommenderNet(num_users, num_movie, 50) 
 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""### Melakukan Training Data"""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 20,
    epochs = 100,
    validation_data = (x_val, y_val)
)

"""## Visualisasi Data Training Model"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""## **Proses Rekomendasi**

### Membuat sistem rekomendasi
"""

movie_df = new_movie
df = preparation
 
# Mengambil sample user
user_id = df.userId.sample(1).iloc[0]
movie_visited_by_user = df[df.userId == user_id]
 
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html 
movie_not_visited = movie_df[~movie_df['id'].isin(movie_visited_by_user.movieId.values)]['id'] 
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)
 
movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

"""### Menerapkan Sistem Rekomendasi"""

ratings = model.predict(user_movie_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_visited[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('----' * 8)
print('movie with high ratings from user')
print('----' * 8)
 
top_movie_user = (
    movie_visited_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
 
movie_df_rows = movie_df[movie_df['id'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print("==================================")
    print("Movie Recomendation : ", row.title)
    print("Genre : ", row.genres)
    print('==================================')
 
print('----'* 8)
print('Top 10 movie recommendation')
print('----' * 8)
 
recommended_movie = movie_df[movie_df['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print("==================================")
    print("Movie Recomendation : ", row.title)
    print("Genre : ", row.genres)
    print('==================================')

"""### Menampilkan history rating terbaik yang diberikan user"""

preparation[(preparation["userId"] == 7) & (preparation["rating"] == 5.0)]