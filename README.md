# Movie-Recomendation

# Laporan Proyek Machine Learning - Ahmad Arbain

## Project Overview

Sektor perfilman menjadi salah satu sektor yang terdampak ketika pandemi Covid-19. Mulai dari terhentinya proses produksi yang melibatkan banyak pekerja seni dan juga penutupan bioskop yang dilakukan untuk memutus penyebaran Covid-19. Untuk itu, Pemerintah telah melakukan evaluasi kebijakan yang ditujukan untuk menghidupkan kembali industri perfilman nasional.

Di masa pandemi Covid-19 ini, pelaku industri di berbagai sektor harus mampu untuk terus berinovasi agar bisa beradaptasi. Menko Airlangga mengatakan bahwa pada masa pandemi Covid-19 industri perfilman terbuka dengan peluang baru yakni berupa layanan streaming berbasis platform digital dengan video on demand. Berdasarkan data statistik, pendapatan dari langganan video on demand Indonesia bisa mencapai US$ 411 juta di tahun 2021 dengan penetrasi pengguna sebesar 16% di tahun 2021 dan diperkirakan akan naik menjadi 20% di tahun 2025. 

> “Layanan streaming ini menjadi peluang tambahan bagi industri perfilman karena dapat menjangkau pasar yang lebih luas bahkan bisa masuk pasar global. Ini peluang besar bagi para sineas Indonesia yang berkiprah di regional maupun global,” lanjut Menko Airlangga.  - *[refrensi](https://newssetup.kontan.co.id/news/airlangga-layanan-streaming-ini-menjadi-peluang-tambahan-bagi-industri-perfilman)*

Berdasarkan permasalahan yang ada maka diperlukannya sebuah inovasi yang dapat dilakukan penyedia layanan streaming salah satunya yaitu melakukan sistem rekomendasi film yang dapat membantu pelanggannya agar terus menikmati layanan mereka guna meningkatkan pendapatan penyedia layanan . Selain itu dalam mengembangkan proyek ini diperlukan sebuah referensi yang dijadikan sebagai acuan rujukan yaitu *[Content Based Filtering](https://repository.its.ac.id/42018/1/2215206701-Master-Thesis.pdf)* dan *[Collaborative Filtering](https://www.kaggle.com/nandha13/movie-recommendation-based-on-genres)*


## Business Understanding
### Problem Statements
- Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan film film lain yang mungkin disukai pengguna dan belum pernah dikunjungi oleh pengguna sebelumnya ? 

### Goals
- Membuat sebuah model machine learning yang dapat menghasilkan sistem rekomendasi berdasarkan rating terbaik user dan rekomendasi berdasarkan film yang belum pernah diberikan rating oleh user menggunakan pendekatan *Collaborative Filtering*


### Solution approach

Solusi yang diterapkan pada pembuatan proyek ini yaitu dengan menerapkan ALgoritma yaitu ***Collaborative Filtering*** dengan ketentuan sebagai berikut :

- ***Collaborative Filtering***. Dalam tahap pengembangan model dengan metode ini yaitu dengan memberikan rekomendasi berdasarkan kemiripan atribut dari item atau barang yang disukai dimana solusi yang diberikan yaitu dengan menerapkan metode Tokenisasi serta mencari nilai similarity dari dataset dengan menerapkan Algoritma *linear karnel* dimana hasilnya akan dijadikan acuan untuk mendapatkan rekomendasi film. kelebihan dari pendekatan user based collaborative filtering adalah dapat menghasilkan rekomendasi yang berkualitas baik. Sedangkan kekurangannya adalah kompleksitas perhitungan akan semakin bertambah seiring dengan bertambahnya pengguna sistem, semakin banyak pengguna (user) yang menggunakan sistem maka proses perekomendasian akan semakin lama.


## Data Understanding

**Informasi Dataset** 

Informasi   | Keterangan 
----------- |------------------
Link        | https://www.kaggle.com/rounakbanik/the-movies-dataset
Lisensi     | CC0: Public Domain
Tags        | earth and nature, movies and tv shows, popular culture
Size        | 943.76 MB

Pada Dataset film yang digunakan mengandung beberapa files yang digunakan sebagai acuan dalam mengembangkan sistem rekomendasi yaitu sebagai berikut :
- *credits.csv* : Dataset yang berisi Cast dan Crew Information untuk setiap film
- *keywords.csv* : Dataset yang berisi tentang seluruh keyword pada film
- *links.csv* : Dataset yang berisi tentang seluruh link pada film  
- *links_small.csv* : Dataset singkat tentang informasi dari link film
- *movies_metadata.csv* : Dataset yang digunakan pada proyek pengembangan model karena mengandung hampir seluruh informasi penting yang digunakan dalam pengembngan model.
- *ratings.csv* : Dataset yang mengandung data ratings yang digunakan dalam pengembangan sistem ini.
- *ratings_small.csv* : Dataset singkat yang mengandung data ratings.



## Data Preparation

pada model development collaborative filtering pada tahap data preparation dilakukan beberapa tahapan yaitu sebagai berikut :
- **Mengatasi Missing Value** 

    Tujuannya dilakukan proses missing value agar dataset menjadi bersih dari fitur fitur yang valuenya kosong karena dapat menimbulkan hasil akurasi yang kurang baik ataupun hasil bias.

    ```
    preparation = preparation.drop_duplicates('movieId')
    preparation
    data_preparation = preparation.dropna()
    ```
- **Membuat Dictionaary Data**
    
    Tujuan pembuatan Dictionary ini agar model yang dibuat hanya memprediksi hasil dari fitur fitur yang hanya digunakan sebagai fitur untuk melakukan proses rekomendasi.

    ```
    new_movie = pd.DataFrame({
        'id': movie_id,
        'title': movie_name,
        'genres': movie_genres,
    })
    new_movie
    ```
- **Melakukan Encoding dan Mapping Data**
    
    Proses encoding dilakukan agar data dapat diubah menjadi data encoding tujuannya agar mempermudah dalam melakukan proses performa pemodelan.

    (a). Melakukan Encoding userId
    
    ```
        user_ids = data1['userId'].unique().tolist()
        print('list userId: ', user_ids)
         
        user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
        print('encoded userId : ', user_to_user_encoded)
         
        user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
        print('encoded angka ke userId: ', user_encoded_to_user)
    ```
    
     (b). Melakukan Encoding movieId
    
    ```
        movie_ids = data1['movieId'].unique().tolist()
        print('list movieId: ', movie_ids) 
        
        movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        print('encoded movieId : ', movie_to_movie_encoded) 
        
        movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
        print('encoded angka ke movieId: ', movie_encoded_to_movie)
    ```
- Membagi Data menjadi 80% Data Training dan 20% Data Test
    
    Proses membagi data menjadi dua bagian training dan testing adalah agar model dapat belajar melalui data training dan dapat melakukan testing dengan data testing. 

    ```
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
    ```
    hasilnya :
    ```
        [[ 127  165]
         [ 664 9064]
         [ 119 2763]
         ...
         [ 531 2199]
         [  11  346]
         [ 101   17]] [1.         0.11111111 0.55555556 ... 0.66666667 0.77777778 0.55555556]
    ```
## Modeling

Dalam tahap pengembangan development model pada studi kasus rekomendasi film ini digunakan dua solusi metode yaitu *Collaborative Filtering* dimana melalui pendekatan solusi masalah ini diharapkan model mampu menghasilkan sebuah hasil rekomendasi berdasarkan hasil permasalahan yang ada. Berikut ini adalah proses tahap development model pada kasus rekomendasi film :

Dalam pembuatan model sistem rekomendasi ini model dibuat dengan menginisialisasi dengan membuat kelas bernama RecommenderNet() berserta paramaternya yaitu (num_users, num_movie, 50) dimana parameter ini maksudnya adalah menyematkan sebuah paramaters user dan pengguna kedalam vektor 50 dimensi dimana model pembuatan kelas ini terinspirasi dari proses tutorial [Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) selanjutnya pada pembuatan kelas pula terdapat proses yaitu model menghitung skor kecocokan antara pengguna dan penyematan film melalui dot product, dan menambahkan bias per film dan per pengguna. Skor pertandingan diskalakan ke interval [0, 1] melalui sigmoid (karena peringkat kami dinormalisasi ke kisaran ini). Hasil akhir dari proses pembuatan kelas ini ialah sebuah model yang dapat di training dengan menggunakan pendekatan loss, optimizer, dan matrik evaluasi yang akan dibahas pada bagian evaluasi.

Tahapan selanjutnya yaitu pembuatan sebuah algoritma untuk melakukan proses rekomendasi dimana proses rekomendasi ini tahapannya yaitu mengambil sampel user secara acak kemudian melakuakan sebuah pencocokan dengan model yang sudah di training dimana selanjutnya dimulai dilakukan pencocokan data film yang belum pernah ditoton user kemudian dilakukan proses rekomedasi film berdasarkan Top rating user dan rekomendasi yang belum pernah ditonton setelah itu hasil pembuatan Algoritma tersebut diterapkan pada sistem rekomendasi. Berikut adalah hasil rekomendasi dari sistem model :

Hasil rekomendasi 
    
```
    Showing recommendations for users: 7
    --------------------------------
    movie with high ratings from user
    --------------------------------
    ==================================
    Movie Recomendation :  Super Mario Bros.
    Genre :  ['Adventure', 'Comedy', 'Family', 'Fantasy']
    ==================================
    ==================================
    Movie Recomendation :  Paris, France
    Genre :  ['Drama', 'Comedy']
    ==================================
    ==================================
    Movie Recomendation :  Cronos
    Genre :  ['Drama', 'Horror', 'Thriller']
    ==================================
    ==================================
    Movie Recomendation :  Little Big League
    Genre :  ['Comedy', 'Family']
    ==================================
    ==================================
    Movie Recomendation :  Wide Eyed and Legless
    Genre :  ['TV Movie', 'Drama', 'Romance', 'Comedy']
    ==================================
    --------------------------------
    Top 10 movie recommendation
    --------------------------------
    ==================================
    Movie Recomendation :  Stonewall
    Genre :  ['Romance', 'Drama', 'Comedy']
    ==================================
    ==================================
    Movie Recomendation :  Michael
    Genre :  ['Comedy', 'Drama', 'Fantasy']
    ==================================
    ==================================
    Movie Recomendation :  Pennies from Heaven
    Genre :  ['Drama', 'Music', 'Romance']
    ==================================
    ==================================
    Movie Recomendation :  Shades of Ray
    Genre :  ['Comedy', 'Romance']
    ==================================
    ==================================
    Movie Recomendation :  Smashing Pumpkins: If All Goes Wrong
    Genre :  ['Documentary', 'Music']
    ==================================
    ==================================
    Movie Recomendation :  Nothing to Declare
    Genre :  ['Comedy']
    ==================================
    ==================================
    Movie Recomendation :  Central Airport
    Genre :  ['Drama']
    ==================================
    ==================================
    Movie Recomendation :  Thani Oruvan
    Genre :  ['Romance', 'Thriller', 'Action']
    ==================================
    ==================================
    Movie Recomendation :  The Legend of Ben Hall
    Genre :  ['Action', 'Drama', 'History', 'Western']
    ==================================
    ==================================
    Movie Recomendation :  The Model Solution
    Genre :  ['Drama']
    ==================================
```
    
    Daftar hasil Top Rating dari penilaian User
    
```
        userId	movieId	rating	title	            genres
    541	    7	745	    5.0	    Super Mario Bros.	[Adventure, Comedy, Family, Fantasy]
    553	    7	1148	5.0	    Paris, France	    [Drama, Comedy]
    559 	7	1223	5.0	    Cronos	            [Drama, Horror, Thriller]
    563	    7	1242	5.0	    Little Big League	[Comedy, Family]
```
## Evaluation
### Collaborative Filtering

Pada tahap evaluasi matrik model ini menggunakan Binary Cross Entropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. root mean squared error (RMSE) adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar. Dalam penerapannya teknik ini lebih sulit diterapkan dibandingkan dengan MSE tetapi memiliki nilai akurasi yang cukup baik dalam penerapannya.

-   Root Mean Square Error (RMSE) Formula :

    ![RMSE](https://1.bp.blogspot.com/-AodtifmdR1U/X-NOXo0avGI/AAAAAAAACmI/_jvy7eLB72UB00dW_buPYZCa9ST2yx8XACNcBGAsYHQ/w320-h90/rumus%2Brmse.jpg)

-   Dimana :
    - At = Nilai data Aktual
    - Ft = Nilai hasil peramalan
    - N= banyaknya data
    - ∑ = Summation (Jumlahkan keseluruhan  nilai)
    
Penerapan dalam metode pengembangan model :
Hasil visualisasi Root Mean Square Error (RMSE)

![Data Visualisasi RMSE](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYkAAAEWCAYAAACT7WsrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dd3xV9f348dc7e0JCCCMEQkA2MgOIe1YQxVq3tWqrxX6/bW1ra6utWu36tV87rK1bUbscaFVUVBTBLXvI3pCwkgAhe79/f3xO4AbIuHAvN7l5Px+P++Cec8859324cN7nM4+oKsYYY8zRRIQ6AGOMMW2XJQljjDFNsiRhjDGmSZYkjDHGNMmShDHGmCZZkjDGGNMkSxLGGGOaZEnCmKMQkWdF5Det3HariJwf7Jj8JSKPicg9oY7DtG9RoQ7AGOMfEbkJuEVVT29uO1X9zomJyIQzK0kYE4ZEJDLUMZjwYEnCtGteVc8dIrJCRMpE5GkR6S4ib4tIiYi8LyKp3rZTRWSViBSJyDwRGeJznNEissTb50Ug7rDvuVhElnn7fiYiI/yM8z4RmSEi//K+40sRGSgid4lIvojkishXfLbv7J3LLhHZISK/EZFIL+bHgIkiUioiRd72z4rIoyIyS0TKgHMOrzITkUu9cygWkU0iMslbf5OIbPbi2iIiX/f/lzDhypKECQeXAxcAA4FLgLeBnwPpuH/jt4nIQOB54Ife+lnAGyISIyIxwGvAP4EuwAzvmIBLIMB04FYgDXgcmCkisX7GeYn3HanAUuBdL75ewK+84zZ4FqgFTgJGA1/BVTGtAb4DfK6qSaqa4rPPdcBvgWTgE98vFpHxwD+AO4AU4Exgq4gkAg8Bk1U1GTgVWObneZkwZknChIO/qeoeVd0BfAzMV9WlqloJvIq7yF4NvKWq76lqDfBHIB53UTwFiAYeVNUaVX0ZWOhz/GnA46o6X1XrVPU5oMrbzx8fq+q7qlqLS0TpwO+9eF4A+opIioh0By4CfqiqZaqaD/wFuKaF47+uqp+qar137r5uBqZ751+vqjtUda33WT0wXETiVXWXqq7y87xMGLMkYcLBHp/3FUdZTgIygG0NK1W1HsjF3cVnADu08ZTI23zeZwE/9qqairwqnt7efscTZ6Gq1vks48WahUtau3y+73GgWwvHz23ms97ApsNXqmoZLoF+x/u+t0RkcItnYjoMSxKmo9iJu/gCICKCu3DuAHYBvbx1Dfr4vM8FfquqKT6vBFV9Pkix5uJKKl19vq+Tqg7zPm9qfv/m5v3PBfofdSdXurkA6AmsBZ48xrhNGLIkYTqKl4ApInKeiEQDP8ZdiD8DPsfV/98mItEi8jVgvM++TwLfEZEJ4iSKyBQRSQ5GoKq6C5gN/ElEOolIhIj0F5GzvE32AJleW0prPQ180zv/CBHpJSKDvUb+S722iSqgFFf9ZAxgScJ0EKq6Drge+BtQiGtEvkRVq1W1GvgacBOwD1f98l+ffRcB3wb+DuwHNnrbBtMNQAyw2vvOl3F3+gAfAKuA3SJS2JqDqeoC4Ju4to0DwIe4klUEcDuupLUPOAv4n4CdhWn3xJ5MZ4wxpilWkjDGGNMkSxLGBIg3gK/0KK+fhzo2Y46VVTcZY4xpUlhN8Ne1a1ft27dvqMMwxph2ZfHixYWqmn60z8IqSfTt25dFixaFOgxjjGlXRGRbU59Zm4QxxpgmWZIwxhjTJEsSxhhjmhRWbRJHU1NTQ15eHpWVh0+KGX7i4uLIzMwkOjo61KEYY8JE2CeJvLw8kpOT6du3L43nbwsvqsrevXvJy8sjOzs71OEYY8JE2Fc3VVZWkpaWFtYJAkBESEtL6xAlJmPMiRP2SQII+wTRoKOcpzHmxOkQSaIlNXX17D5QSWVNXcsbG2NMB2JJAqirV/JLKqkKUpIoKirikUce8Xu/iy66iKKioiBEZIwxrWNJAmiopAnWk1aaShK1tbXN7jdr1ixSUlKa3cYYY4Ip7Hs3tUZDXX6wJju888472bRpE6NGjSI6Opq4uDhSU1NZu3Yt69ev56tf/Sq5ublUVlbygx/8gGnTpgGHphkpLS1l8uTJnH766Xz22Wf06tWL119/nfj4+KDEa4wxDTpUkrj/jVWs3ll8xHoFyqtqiYmKIDrSv8LV0IxO/PKSYc1u8/vf/56VK1eybNky5s2bx5QpU1i5cuXBrqrTp0+nS5cuVFRUMG7cOC6//HLS0tIaHWPDhg08//zzPPnkk1x11VW88sorXH/99X7Faowx/upQSaIpJ7pP0Pjx4xuNZXjooYd49dVXAcjNzWXDhg1HJIns7GxGjRoFwNixY9m6desJi9cY03F1qCTR1B1/vSordxygR6c4unWKC3ociYmJB9/PmzeP999/n88//5yEhATOPvvso451iI2NPfg+MjKSioqKoMdpjDHWcI1Pw3WQnr+UnJxMSUnJUT87cOAAqampJCQksHbtWr744ovgBGGMMcegQ5UkmiIiiAhKcLJEWloap512GsOHDyc+Pp7u3bsf/GzSpEk89thjDBkyhEGDBnHKKacEJQZjjDkWYfX40pycHD38oUNr1qxhyJAhLe67aucBUhNiyEhp3z2GWnu+xhjTQEQWq2rO0T4LenWTiEwSkXUislFE7jzK57eLyGoRWSEic0Qky1ufJSJLRGSZiKwSke8ENU6E+jBKmMYYEwhBTRIiEgk8DEwGhgLXisjQwzZbCuSo6gjgZeD/vPW7gImqOgqYANwpIhnBijVCwHKEMcY0FuySxHhgo6puVtVq4AXgUt8NVHWuqpZ7i18Amd76alWt8tbHBjtWEbEkYYwxhwl2kugF5Pos53nrmnIz8HbDgoj0FpEV3jH+oKo7D99BRKaJyCIRWVRQUHDMgYpg1U3GGHOYNtMFVkSuB3KABxrWqWquVw11EnCjiHQ/fD9VfUJVc1Q1Jz09/Zi/P0IIUt8mY4xpv4KdJHYAvX2WM711jYjI+cAvgKk+VUwHeSWIlcAZQYoTQYI2d5MxxrRXwU4SC4EBIpItIjHANcBM3w1EZDTwOC5B5PuszxSReO99KnA6sC5YgbrqpuAc+1inCgd48MEHKS8vb3lDY4wJgqAmCVWtBb4HvAusAV5S1VUi8isRmept9gCQBMzwurs2JJEhwHwRWQ58CPxRVb8MVqwREryShCUJY0x7FfQR16o6C5h12Lp7fd6f38R+7wEjghvdIRLELrC+U4VfcMEFdOvWjZdeeomqqiouu+wy7r//fsrKyrjqqqvIy8ujrq6Oe+65hz179rBz507OOeccunbtyty5c4MToDHGNKFjTcvx9p2w++iFke61ddTXK8T4+VfS42SY/PtmN/GdKnz27Nm8/PLLLFiwAFVl6tSpfPTRRxQUFJCRkcFbb70FuDmdOnfuzJ///Gfmzp1L165d/YvLGGMCoM30bgo14cT0bpo9ezazZ89m9OjRjBkzhrVr17JhwwZOPvlk3nvvPX72s5/x8ccf07lz5xMQjTHGNK9jlSSaueMv3F/OgYpahmZ0CmoIqspdd93FrbfeesRnS5YsYdasWdx9992cd9553HvvvUc5gjHGnDhWkvCIBG/uJt+pwi+88EKmT59OaWkpADt27CA/P5+dO3eSkJDA9ddfzx133MGSJUuO2NcYY060jlWSaEYwB9P5ThU+efJkrrvuOiZOnAhAUlIS//rXv9i4cSN33HEHERERREdH8+ijjwIwbdo0Jk2aREZGhjVcG2NOOJsq3LOnuJI9xZWc3KszIif6gaaBY1OFG2P8FdKpwtuLhrwQRjnTGGOOmyUJj3gPMbVJ/owx5pAOkSRaU6UW0VCSCHIswRROVYfGmLYh7JNEXFwce/fubfEC2tAO0V4vtKrK3r17iYuLC3UoxpgwEva9mzIzM8nLy6OlZ02UV9eyr6wGimKJjmyfuTMuLo7MzMxQh2GMCSNhnySio6PJzs5ucbt3Vu7iOzOXMOu2MxgS5AF1xhjTXrTPW+YgiI2KBKCqti7EkRhjTNthScITE+X+Kqpr60MciTHGtB2WJDyxXpKosiRhjDEHWZLwNFQ3WUnCGGMOsSThibGShDHGHMGShKehuqm6zhqujTGmgSUJz8GSRI2VJIwxpoElCY81XBtjzJFalSREJFJE/hjsYELJusAaY8yRWpUkVLUOOD3IsYSUDaYzxpgj+TMtx1IRmQnMAMoaVqrqfwMeVQhER7oJ/qwkYYwxh/iTJOKAvcC5PusUCIskISLERkVYm4QxxvhodZJQ1W8GM5C2IMaShDHGNNLq3k0ikikir4pIvvd6RUTCal7q2KhISxLGGOPDny6wzwAzgQzv9Ya3LmzERkVYm4QxxvjwJ0mkq+ozqlrrvZ4F0oMUV0i4Ngnr3WSMMQ38SRJ7ReR6b8xEpIhcj2vIDhsxVpIwxphG/EkS3wKuAnYDu4ArgLBqzLbeTcYY01irejeJSCTwO1WdGuR4Qso1XFt1kzHGNPBnxHWWiMQEOZ6QsuomY4xpzJ/BdJuBT71R174jrv8c8KhCJDYqgv3lliSMMaaBP20Sm4A3vX2SfV7NEpFJIrJORDaKyJ1H+fx2EVktIitEZI6IZHnrR4nI5yKyyvvsaj9iPSZWkjDGmMb8aZMYqKpf9+fg3n4PAxcAecBCEZmpqqt9NlsK5KhquYj8D/B/wNVAOXCDqm4QkQxgsYi8q6pF/sTgD2u4NsaYxoLdJjEe2Kiqm1W1GngBuPSwY89V1XJv8Qsg01u/XlU3eO93AvkEeVyGlSSMMaaxYLdJ9AJyfZbzgAnNbH8z8PbhK0VkPBCDq/I6/LNpwDSAPn36NHPollnvJmOMacyfJLHJezW0SQSUNzgvBzjrsPU9gX8CN6rqEbf5qvoE8ARATk6OHk8MVpIwxpjG/JkF9n4AEUnwqR5qyQ6gt89ypreuERE5H/gFcJaqVvms7wS8BfxCVb9obazHytokjDGmMX9mgZ0oIquBtd7ySBF5pIXdFgIDRCTba8+4BjdJoO9xRwOPA1NVNd9nfQzwKvAPVX25tXEej5ioCGrrlbr64yqQGGNM2PCnC+yDwIV48zWp6nLgzOZ2UNVa4HvAu8Aa4CVVXSUivxKRhtHbDwBJwAwRWea1eYCbAuRM4CZv/TIRGeVHvH5reISpVTkZY4zjT5sEqporIr6rWmzlVdVZwKzD1t3r8/78Jvb7F/Avf+I7XrFRLmdW1dYRHxN5Ir/aGGPaJH+SRK6InAqoiEQDP8CVDsJGjJckrCRhjDGOP9VN3wG+i+vWugMY5S2HjUMlCUsSxhgD/vVuKgSaHHEtInep6v8LSFQhEmNJwhhjGvGnJNGSKwN4rJBoaLi2AXXGGOMEMklIy5u0bbHWJmGMMY0EMkm0+8EF1iZhjDGNWUnCh/VuMsaYxgKZJGYE8FghcahNwpKEMcZAK3o3icjfaKYqSVVv8/78XQDjCgkrSRhjTGOtKUksAhYDccAYYIP3GoWbvjts+I64NsYY04qShKo+B+A9Ne50bz4mROQx4OPghndi2TgJY4xpzJ82iVSgk89ykrcubFgXWGOMacyfuZt+DywVkbm4nkxnAvcFI6gTrrYa9m0mNrYbYNVNxhjToNUlCVV9Bvfo0VeB/wITG6qi2r381fDIBGJzXe2ZlSSMMcbx56FDApwPjFTV14EY79nT7V9qXwCiirYiYm0SxhjTwJ82iUeAicC13nIJ8HDAIwqF+BSIT0WKthITac+5NsaYBv60SUxQ1TEishRAVfd7jxgND6nZsG+LPefaGGN8+FOSqBGRSLyBdSKSDoTP1TS1L+zfQkxUpCUJY4zx+JMkHsI1WncTkd8CnwDtfpT1QV2yoSiXhEi13k3GGONpVXWTiEQAW4CfAufhusB+VVXD5/GlqdmgdfSO3Et1bVqoozHGmDahVUlCVetF5GFVHQ2sDXJMoeH1cOojeyisPSm0sRhjTBvhT3XTHBG53OsKG366ZAPQm3zr3WSMMR5/ksStuOnAq0SkWERKRKQ4SHGdeMkZEBlLL3Zbm4Qxxnha3QVWVZODGUjIRURAahYZZbutd5Mxxnj8GSeBiKQCA3DThgOgqh8FOqiQSe1L9+JNVt1kjDGeVicJEbkF+AGQCSwDTgE+B84NTmghkJpN+qZPqKqx6iZjjAH/2iR+AIwDtqnqOcBooCgoUYVKl2zi68uJrw2v0zLGmGPlT5KoVNVKABGJVdW1wKDghBUiXjfY9JqdoY3DGGPaCH/aJPJEJAV4DXhPRPYD24ITVoikum6w3et2hTgQY4xpG/zp3XSZ9/Y+78FDnYF3ghJVqKRmAdCzbneIAzHGmLbBn4brPj6LW7w/ewDbAxpRKEXHUxKdTkblnlBHYowxbYI/1U1v4WaAFVwX2GxgHTAsCHGFTHF8Jr2r9lBXr0RGhOfgcmOMaS1/qptO9l0WkTHA/wY8ohArScikz4FPqa6tJz4mMtThGGNMSPnTu6kRVV2Ce+Z1s0RkkoisE5GNInLnUT6/XURWi8gKEZkjIlk+n70jIkUi8uaxxumvsoQ+9JR9VFWUnqivNMaYNsufNonbfRYjgDFAs31FvYcUPQxcAOQBC0Vkpqqu9tlsKZCjquUi8j/A/wFXe589ACTg5o06ISqSMgGo3bcVOo88UV9rjDFtkj8liWSfVyyujeLSFvYZD2xU1c2qWg28cPg+qjpXVcu9xS9wI7obPpuDe5b2CVOV7Aoyum/rifxaY4xpk/xpk7j/GI7fC8j1Wc6j+Sqqm4G3/fkCEZkGTAPo06dPC1u3rDqlH/UqROxeTss50Bhjwps/1U1v4D3f+mhUderxBCIi1wM5wFn+7KeqTwBPAOTk5DQZX6vjiE9lhfZj4La5wL3HezhjjGnX/OkCuxk3LuJf3vK1wB7cCOym7AB6+yxneusaEZHzgV8AZ6lqlR8xBVxsdATz6kcyMv9VKNsLifYoU2NMx+VPm8Rpqnq1qr7hva4DzlDVD1X1wyb2WQgMEJFsEYkBrgFm+m4gIqOBx4Gpqpp/LCcRSLGREcytG4WgsOmDUIdjjDEh5U+SSBSRfg0LIpINJDa3g6rWAt8D3gXWAC+p6ioR+ZWINFRPPQAkATNEZJmIHEwiIvIx7ml454lInohc6Ee8xyQ2OoIV2o/q2FTY+F6wv84YY9o0f6qbfgTME5HNuFHXWXgNxs1R1VnArMPW3evz/vxm9j3Dj/gCIiYyEiWCvd3PoOfG96G+3j21zhhjOiB/eje9IyIDgMHeqrWhbj8IhtholxB2dT+Dnttnws6lkDk2xFEZY0xotPoWWUSuBGJUdTlwCfC8NzVHWImJdH8lO7tMBMSqnIwxHZo/9Sj3qGqJiJwOnAc8DTwanLBCJy7azddUJJ2g11jYYEnCGNNx+ZMkGh78PAV4UlXfAmICH1JodU2KITpSyN1fDgO+AjsWQ1lhqMMyxpiQ8CdJ7BCRx3HzKs0SkVg/928XoiIjyEpLZEtBGQw4H1DY+H6owzLGmJDw5yJ/Fa4r64WqWgR0Ae5o+FBEUgMcW8j065rI5sIy6DkaUvrAwqdBj3swtzHGBM7at2DLR0G/NvnTu6kc+K/P8i7A92HQc3Azw7Z72emJzF2XT61C1MTvw9t3wPbPIevUUIdmjDkWqq59sfd4iE8JdTSH1NVA7nxY97b7M20A9D0Nsk6DLtlN7zf/CXddAsg6Hc79RdCuT/6Mk2hJ2DzGrX/XJGrqlB1FFWSNvh4+/D188qAlCWPaq4/+CHN/A4np8JXfwIirQQSKd8HWjyG1L/TKCdyYqNpqyF8NcZ0goStExcL+bbBvE+zdCAXroHCD26aqGCJjoOco2PAuLP+PO0b2WXDOz6HPKY2PvfhZlyAGTYHsM+GTP8Mzk905fe2JwMTvI5BJImzqY/qlu4HkmwvKyErrBuNvhXm/gz2rofvQEEdnjPHLmjddghh8MZTshldvhYVPQV017Fp+aLtOvWDIVNcWmTneXeCbU77PVfeAO3akdzkt3AgzboI9Xza9b0JXSB8EJ18J/c+BfmdDbLIbvFu4Dta/A58/DNMvhP7nQt/TIa6z+865v3Odaq58xiWfMTfAoqfd50EgGqD6LBFZoqohrW7KycnRRYsWHfdx9pVVM+bX73H3lCHcckY/98P8ZRgMvRQueywAkRoTJKX57uKXnNHyXXFdDWz7FA7kQfpg6DYEYg6baafyAKx8BXIXuIvV4ClHbhNIqvDlDPe9OTc3PoeqkkOxSisrLvashqcvcBfkm2a5O/al/3Ali069YOCF7iJdsB5Wv+46qdRVgURA92FuG1XQercuMtodY/8W2LmMg/fGXfrBmXeARMJbt7ttzv8lRERDeSHUVEJqFnTpD2n9IaFLy7FXl7lk9tnfocxnWrt+Z8O1L0J0XCv/UlsmIotVNeeonwUwSSxV1dEBOdgxClSSUFVG/eo9pozoye8u8x7t/fadsPBJuG0ZpPRu/gDGnEjV5bD2TVj2H9g8D1CIioPUbHeh63OKqyqNS4G9G1w1x7bPYOMcqDrgcyBx/7ZTsqBzJtRWwbpZUFsJMclQXQLRiTB0Kpx/HyT3COx5FG2HN34Im+a45f7nueqTxK6wdpa7+Jbsgq4DYfT1MOxr0CkDIiKhrha2zIMVM1z1UWwnSOrmzlXrYdo86NSz5RiqSiFvIWz/AnK/cDeIEuGSkta7xFpX46qt+p3tEkxpvquS3u2VHHqfAldMh869Avd3U1PpEmd1qftdAzxVUMCShPc40u74VFOp6nbvsy6quu84Yz0ugUoSAJc98ilxUZE8P82rDyzaDg+Ndv9wr/4XRIXdEBETKjuXwge/cVUcgy+C7sNbf6dcuAGemwolO11PvJHXuov3vs2wd5M7dsmuI/dL7Obuogdd5C66BWtd/XjBOne3fiDPJYehl7oLcs9RrvPGihdhxUuuKuaqfxxZX340VaUuaUUepXa7utxdlLd8CPMfd3ftF9zvLszv3OXuuHuNdUmw+3AY/Q1Y9aq7gIPbLiEN6muhYj/EdnbVRbVVh0pVU/4c/Kl1VF1CLcqFcbcc/VzbsIAkCRH5PvBL3DMk6r3VqqojAhJlAAQySfz4peV8srGA+T/3mX9w4VPw1o9dg9GVz1qi6MiKd7ouiFGxMPzyY6+CKVgH0ye5u9PqUkBdI+rl01u+sBWsh+cudne4X3vSNXQefoepCkXbYNvnUFPmEkLXgZDUvfWJ6HB7VsGL17sbp/PvcyWPvRvdco+TXfLpnAn5a+DTh+DLl1xSGnODe9VVuwvq2lmuR099DSCunn3KH12yA9de8NKNULwDzvopnPZDV90DLjlumgtlBa4qpq4GBk12x4iKPbbz6sAClSQ2AhNUdW8ggwukQCaJh+du5IF317Hy/gtJivW5K2joejb4YpcoGv7RBlrFftfgFhHpLkCJ3aD3BJuRNtSWv+gaCXPnH1oX19ld/Mbf6l9VZNH2QwniW++4hst1b8PHf4LKIrjxDeg58uj7FqyDZy927298A7oNPvp2wVJR5BqA179zaF1DlRS4KpH9WyAqHkZd6851o1eN1FCP322Yu+vPOr3prqnV5a73T6CrtkwjzSUJf8pEucCBFrcKE/26ujvDrYVlDO/l02tgwjTQOnjnTnjh667uMTYpcF9cVwOLpsO8/+cSha+cm2HKn479DtAcu7oa95svfArSh8A5d8Owr0L5Xpj/GHz+CCx4yvVXP+V/XXIHKNnj6sh3r3B11iW7XWNoSh/XflBVCt+c5RozAcbe6Oq6n7kI/nkZ3PSWa1AGd8HcNAfWvOHuwmMS4MY3IX3gif/7iE+Ba5535xbXyTXIxia75LXhXdjyMYy8BsZ9+9DTHfdvdVVVMYmumqu5cQANYhLcy4SMPyWJp4FBwFvAwSnCVfXPwQnNf4EsSazbXcKFD37EX68ZxaWjjtIAtfBpmPUTV0963YuuAe147d0Ez18Dhetd/+fz7nN1stWlsOx5+OJhV+S+4P4j91V1dc9J3Q9doFqy5g1XbTLymqB1n/NLxX7IW+QuIr0ntP48jkV1OSx/3l20uvSDk86DfuccuqA1/L8QcXHNuMld1E/9Ppx//5GxFW2HWT+F9W9DxhjXZ33tm7D1E0Bdb5f0wa4apniH6zMvEXDt80ev19+7ySWK+lpXPXQgz+2ndRDfxV1kz7j9UHIx5jgEqiSx3XvFEIYT+x0uKy0BETdW4qjG3ezuBmfcBE+eC2f82P2HrqmAmCRI6+cuPkndAXEXm8jYpquLVGHmbVC6x92hDZrcuMRw4XCorYBPH3Qll4GTXUPjnpWu7nbXcncxO/lKVz/dXGmjsti1rXz5klue82vIueno1SXFO92xo+NddULnXoEt+tfVwEcPwOqZULDm0PqENHeOY2+C3uOO/3vq6+FArrvTzZ0Pi59xpYD0Ie7C3jCASSJcHX8D8X4viYRLH3aNuEeT0sdd8Fe+Am//FN75GaSd5OrSB0+BbkOPrJps7oFWaf3hhtfhjR+43zJrInTuDdlnuOqZdtYwatqvgHWBbQsCWZIAOP0PHzCmTyoPXdtMz97dK+E/V0NxXssHTOgKE251vR8O7ye9/EV4dRpc/BfI+dbR96+vd/XADRd3cP2wuw2BjFFQXwfL/g0X/g4mfvfox9j2Obz2HXfne9adMOACN2hn1avuLrXnSBg4yXXxW/Wa60fvO05SIt0gnqGXtny+tdXw6V9d//ABF0D8YdN7lexxSXb7Z66Kpe/prgRRvs81Cq9/1zW2XvLXpi/OLdm7Cb54xP39NtSXg7sTn/g91zVU610voC0fQU25lxjEnbfWu9egKa3vIVNR5BpU006yqkHTLgSq4Tod+CkwDDg4ikNVzw1EkIEQ6CRxw/QF7Cur4s3vt/AU1doqd2GLinVd/aqKD3VBLN+Lu9io63u94V3X13zct+Csn7l63Ioi+Ps4dxd/83vNV7PU1bhqkugEd3fadcChO1RV1+tk3dtww2uuyqph/bZP3R375nnujvTypxpXc+zf5u6C178LeQvchbHrQBh+hesL3tD75qM/ugvqdS/ASV7Pr/p61wWzU69DF0VVmPl9WPpPt/+9E+QAAB5HSURBVBwR5ZJAz1EuacQkwXv3unO/9O9w8hVHnmtlMbx0A2ye6xLa2Xe2/qKbu9CVuta+5b57+OXufNMHu4FVrRnMZEwHEagkMRt4EfgJ8B3gRqBAVX8WqECPV6CTxH0zVzFjUS4r778QCdQd4Z5V7u56xUvuonrxn12vjwVPwLS5kHGc4xEri+Gp81zSOuPH7vt2LHZVOYndXJ16zreab2wv2wsV+45+J1xR5LpdFm6Eq//pkuH8x92cNEMvhUseco2an/0NZt/tYhh0kaufX/eO6ypZX+OOldoXrv439BjedCx1NW6A1bJ/uTEqgyZDn4mupLNjsetjX7LbXfh7DHelqU8fgm2fuMFj426G8dOsd4wxzQhUklisqmNFZEXD2AgRWaiqAagwDoxAJ4l/fr6Ve15fxfyfn0f3ToEbAg+4aQ5mft8NYgLXC2TKHwNz7IL1rp2kusRVcWWMggEXwphvuLaF41Va4CYU27vBLWeOc6/5j7s2i5xvwfv3u6RxxTON693r69xFvXiHqyaLTW75+1RdqWDBk24/XxFRrv2idM+hdZ16uaqkMTcEtueZMWEqUA3X3u0fu0RkCrAT90yJsJXd1V1gNhWUBj5J9B4Pt34Mn/zFdSM89+7AHTt9IHxvIaCQ3DPw9eJJ6a5RdeGTMPiSQ3X1wy6Dl2+G9+9zPXy++uiRDbMRkS6R+DNlgQic/iPXs6tou5tSonyvG4mbMcolvvJ9riG/sthVg9lAR2MCwp+SxMXAx0Bv4G9AJ+B+VZ0ZvPD8E+iSxM6iCk79/Qf85qvDuf6UrIAdN6xV7IfFz8Go69zcOcaYNi8gJQlVfdN7ewA4JxCBtXU9O8eRkhDNirwiwJJEq8Snwuk/DHUUxpgAafUcDyIyUETmiMhKb3mEiASwjqTtERFysrqwcOv+ljc2xpgw5M9EQE8Cd+G1TajqCuCaYATVlkzI7sKWwjLySypDHYoxxpxw/iSJBFVdcNi62kAG0xaNy3Zt8wu3WGnCGNPx+JMkCkWkP97wWxG5AjjKRPXhZVhGJxJiIlmwpc1OfmuMMUHjTxfY7wJPAINFZAewBTjGuRLaj+jICMb0SWWBtUsYYzqgVpckVHWzqp4PpAODVfV0Vd0atMjakPHZXVi7u5gDFTUtb2yMMWGk1SUJEUkBbgD6AlEN01So6m1BiawNGde3C6qweNs+zh3cPdThGGPMCeNPm8QsXIL4Eljs8wp7o/ukEB0pzN8S0kd4G2PMCedPm0Scqt4etEjasLjoSEZkprDQkoQxpoPxpyTxTxH5toj0FJEuDa+WdhKRSSKyTkQ2isidR/n8dhFZLSIrvMF6WT6f3SgiG7zXjX7EGnDj+nZhRd4BKqrrQhmGMcacUP4kiWrgAeBzDlU1NTtRkohEAg8Dk4GhwLUiMvSwzZYCOd7Msi8D/+ft2wX4JTABGA/8UkQOe2rNiTMhuwu19crSXOvlZIzpOPxJEj8GTlLVvqqa7b36tbDPeGCj1zOqGngBaPRIM1Wdq6rl3uIXQKb3/kLgPVXdp6r7gfeASX7EG1Bj+6YiAp9vsvESxpiOw58ksREob3GrxnoBuT7Led66ptwMvO3PviIyTUQWiciigoICP8NrvU5x0Uzsl8bM5TsJp0e+GmNMc/xJEmXAMhF5XEQeangFKhARuR7IwVVptZqqPqGqOaqak56eHqhwjuqKsZls21vOAmvANsZ0EP4kideA3wKf0fousDtwz59okOmta0REzgd+AUxV1Sp/9j2RJg3vQVJsFC8vzgtlGMYYc8L48zyJ55r7XEReUdXLD1u9EBggItm4C/w1wHWH7TcaeByYpKr5Ph+9C/zOp7H6K7hZaEMmISaKKSf35I0VO7lv6jASY/3pQWyMMe2PPyWJlhzRiK2qtcD3cBf8NcBLqrpKRH4lIlO9zR4AkoAZIrJMRGZ6++4Dfo1LNAuBX3nrQurKnEzKq+uY9WXYz21ojDF+DaZryVFbc1V1Fm60tu+6e33en9/kAVWnA9MDFWAgjM1KJbtrIi8vzuPKnN4t72CMMe1YIEsSHYKIcMXYTOZv2cf2vf529jLGmPYlkElCAnisNu1rY3ohAjMW57a8sTHGtGP+POP6By2s+1lAImoHenaO59xB3fj3/O02TYcxJqz5U5I42txJNzW8UdXZxx1NOzLtzH7sK6vm5SXWHdYYE75abLgWkWtx3VazG3oeeZKBkPc2CpXx2V0YmdmZpz7ezHXj+xAZ0WFq24wxHUhrejd9hnuWdVfgTz7rS4AVwQiqPRARpp3Zn+/+Zwnvrd7NpOE9Qx2SMcYEXIvVTaq6TVXnqepEYC2uBJEM5HnjIDqsScN70KdLAo9/tNnmczLGhCV/Gq6vBBYAVwJXAfNF5IpgBdYeREYIt5yRzdLtRSzaZlOIG2PCjz8N13cD41T1RlW9ATcN+D3BCav9uHJsb1ITorn39VXkl1SGOhxjjAkof5JExGFzK+31c/+wFB8TyV+uHsXWwjIue/gzNuwpCXVIxhgTMP5c5N8RkXdF5CYRuQl4i8Om2+iozh7UjRdvPYWq2nouf/Qz5m+2BxMZY8JDq5OEqt6Bm611hPd6QlU7zAC6lozITOHV/z2VtKRYfvTiMmrr6kMdkjHGHDd/q4s+BeYCH3jvjY/eXRL4+UVD2HmgkndX7Ql1OMYYc9z86d10Fa530xVY76YmnTu4G326JDD90y2hDsUYY46bPyWJX2C9m1oUGSHceGpfFm/bz4q8olCHY4wxx8V6NwXBVTmZJMVG8cynW0MdijHGHJdWXeRFRICF1rupdZLjorlibCZvrthJfrGNnTDGtF+tShLq5pwYj/VuarWbTu1Lbb3yzy+2hToUY4w5Zv48vnQxkKuqtwcrmHDSt2siXxnanUfmbSJChO+fexJRkVY7Z4xpX/y5ak0APheRTSKyouEVrMDCwQNXjmTqyAz+OmcDVz3+Odv2loU6JGOM8Yu0dvZSEck62npVbTP1KTk5Obpo0aJQh3GE15ft4O7XVlJRXcfkk3tyw8QscrJScU09xhgTWiKyWFVzjvZZq6ub2lIyaG8uHdWLcX278NTHW5ixOJc3lu9kVO8UnvjGWLp1igt1eMYY0ySrJD9BMlLiufeSocz/+Xn89rLhrN9TwlWPf86OoopQh2aMMU2yJHGCJcRE8fUJWfzz5gnsLavmqsc+Z0uhtVUYY9omSxIhMjYrlee/fQoVNXVc9fjn5O4rD3VIxhhzBEsSITS8V2demHYKVTV13PLcIkoqa0IdkjHGNGJJIsQGdk/m0evHsqmglO/9Z+nBKcaLyqtZsn0/9fX27GxjTOj4M5jOBMlpJ3Xl118dzl3//ZLv/mcJFTX1fLaxkNp6ZcqInvzpypHERUeGOkxjTAdkSaKNuHZ8H7YUlvHER5vp0yWBm8/IJjYqkofmbCC/uJInvpFDamJMqMM0xnQwrR5M1x601cF0raWq7CiqoFdK/MGBdm+u2MntLy0nMyWef397Aj07x4c4SmNMuGluMJ21SbQhIkJmakKjkdgXj8jg37dMIL+kipufXURZVW0IIzTGdDSWJNqBcX278PfrRrN2dzE/eGEpdT6N2cWVNda4bYwJGmuTaCfOHtSN+6cO457XV3H/G6sY1COZV5fsYNG2/eRkpfKHK0bQPz0JgJq6ej5aX8Dgnp3olWLVU8aYYxf0JCEik4C/ApHAU6r6+8M+PxN4EPeMimtU9WWfz/4ATPEWf62qLwY73rbsGxP7srmw7OAT7wZ0S+KW07OZsTiPyX/9mNvOPYnKmnpeXJRLQUkVg7on88b3TycmygqMxphjE9QkISKRwMPABUAe7ul2M1V1tc9m24GbgJ8ctu8UYAwwCogF5onI26paHMyY27q7pwxlSI9ODM3oxLCMTogI087qx72vreKPs9cjAucO6sao3in86b31/P2DDdz+lUGhDtsY004FuyQxHtioqpsBROQF4FLgYJJQ1a3eZ/WH7TsU+EhVa4Fa79kVk4CXghxzmxYZIVw1rnejdd2S43jsG2NZlltE16QYMlMTANhSWMYj8zbxlWE9GN6rcyjCNca0c8Guh+gF5Pos53nrWmM5MElEEkSkK3AO0PvwjURkmogsEpFFBQUFxx1wezaqd8rBBAFw7yVDSU2M4Y6XV1Bde3gONsaYlrXZympVnQ3MAj4Dngc+B+qOst0Tqpqjqjnp6eknOMq2LSUhht9ddjJrdhXzh3fWEk5jYowxJ0awk8QOGt/9Z3rrWkVVf6uqo1T1AkCA9QGOL+xdMLQ7N0zM4ulPtvCzV1ZQU3eoRKGq1n3WGNOsYLdJLAQGiEg2LjlcA1zXmh29Ru8UVd0rIiNwvZ9mBy3SMHb/1GGkJMTw0JwN7Cmu4pYzspm9ag/vrNpNTV09k4f3ZOrIDMZndyFCoK5eiRAhIsIer2pMRxf0aTlE5CJcF9dIYLqq/lZEfgUsUtWZIjIOeBVIBSqB3ao6TETigCXeYYqB76jqsua+q71PyxFszy/Yzi9e/ZJ6hbjoCM4d3I3oyAjeW72H8urGNXmd4qK4ZGQGV4zNZFTvFHsetzFhrLlpOWzupg5meW4Ruw5UcObAdBJiXEGyorqOOWv3sG53CREiREYImwtKeWfVbipr6hncI5m7LhrCWQOtzceYcGRJwhyT4soa3lqxi8c/3MTWveWcP6QbP7pgIIJQUFpFeVUtg3t2om9agpU0jGnHLEmY41JVW8f0T7by9w82UFZ9RAczUhOiGZuVyv+c3Z+xWV1CEKEx5nhYkjABsae4krlr8+kcH016ciyxUZGs2nmAJdv3M3ddAQUlVVw2uhd3Th5M905x1NUr5dW1JMVGWUnDmDbMkoQJurKqWh6Zt5EnP9oCArFREZRUumnNE2Ii6ZuWSHZ6Ir1TE+iVGk9mSjxjslLpHB/d6Dgb9pRQUFJFj85x9OwcT2xUBBU1dZRX1xEfE0lSbNufk3JfWTX5JZUM7tEp1KEY0yqWJMwJs21vGc9+thVV6BQfTWJMJLuLK9lSWMaWwjJ2FlVQU+f+zSXERHLl2ExuPLUvuw5U8tiHm/h4Q2GTxxaB/ulJjOqdwri+qZw9qBvdO8W1GNPWwjJeW7aDnKwuTOyfRmQQu/bW1tVz6cOfsiG/lPd+dCZZaYlB+y5jAsWShGkz6uqVgpIqthSWMWNxLm8s33kwaaQnx3LTqX0Z3TuF3cWV7DpQSVVNHYmxUSTERrG/rJpluUUsyy1iX1k1AMMyOnHRyT25xXvcq6+q2joe/3Azf5+78eC0JOnJsUwdmcENE7OavYBvzC+hU1w03VqRhHxN/2QLv3pzNVERwmkndeXZb46zqjbT5lmSMG1Wfkkl/12ygy4JMVw6OuOIC/3RqCrr95Tywdp85qzZw6Jt+xncI5k/XzWKoRmdqKyp460Vu3h43kY2F5Rx8Yie/PTCwazceYDXlu5g7rp86uqVqSMz+N9zTmJg9+SDxy4qr+a3b61hxuI8RGBCdhcuHpFBj05x7C2rorC0mpjICHqmxJGREs+g7skkelVguw9Uct6f5pHTtwtnDUznV2+u5pGvj+Gik3sG7e+vKVW1dXyyoZCeneMZmmHVXqZ5liRMWJu7Np+fvrKCovJqJg3vyUfrCzhQUUP/9ETuuXgoZw/q1mj7/OJKnvx4M/+ev53y6jr6pycyNiuVrLREnvl0C0XlNdxyRj9ioyJ4Y8VONheUNfndXRJj+NEFA7l2XG9ue2Epc9bkM/tHZ9IrJZ5LH/6UwtIq3r/9LJLjGre9bN9bzrurdtM5IZruneLo0yWB7K5Hlmy2FpYRGSF0TYolPqblBLp0+37+M38776zaTUllLQkxkTz7zfGMzz7U6yy/uJKSqtqDD6k6mo83FLBw636uHte7zTy4qqyqli2FZQzt2emEzAawaucBuibFtqpK83Dr95Tw2cZCrj8li6jINjtF3kGWJEzY21dWzd2vfcn7a/K5YGh3vj6hDxP7pTVb1bO/rJqXFuWyYMs+Fm/fT1F5DSMyO/P7r404ePfdUGqprKmja3IsaYkxVNXWs7Oogtx95Tz9yRbmb9lHny4JbN9Xzk++MpDvnTsAgGW5RVz2yKdcNbY3d0waRNek2INVYA/P3UjVYTPzTh7eg7svHkqvlHh2FFXwmzdX8/bK3Qc/T4qNIjM1nn7piWR3TWTqyF4M6nGoFPTPL7Zx38xVJERH8pVhPbhgaDceeHcduw5U8ty3xjO2Tyr/mr+NP7y9lpp65fFvjOWcwxLout0l/G7WGj5c72ZUjo2K4JYzsvmfs09q1Gmgrl6ZsSiX5xds55R+adx4al8yjpJMCkurePLjzWzYU0pBSRV7S6vo3y2Jr43pxYXDehwc0NmSksoarn9qPsvzDtC9UyyThvXg7EHdSEuKITnOtX3VK9TW11NT53rVlVfXsa+smuW5RSzetp8thWXcdt4Arj8lq9nvWrBlH3/7YAMfbygkLTGGJ27IYWxWaqviBJi/eS+3PLeIkqpaLhzWnb9eM5q46OYT/LLcIh6bt4mUhGhO6pbEgO7J9OwcR3pSLCkJ0UGvsrQkYToMVT2m/1Cqyu7iSrolx/nVsK2qvLtqN7+dtYbEmChmfq/xkwDvm7mKZz/bCkD/9ERq6pTt+8qZcnJPfjZpMCKua/GnG/fy6IcbAZhycgazvtyFonznrP5kdI6noLSKgpIqtu8rZ0thGbn7yqlX5ZrxffjheQN49MNNPPPpVs4f0o0Hrxl98IKeX1zJNU9+wZ4DlQzonsyy3CLOGNCVfWXVbNhTyt+vG81XhvUgd185D83ZwCtL8kiKjeK28wZw/pDuPPj+el5btpPUhGjOGpjOqSd1JT05lj/NXsfKHcX065rItn3lAEwa3oNzBnVjeK9OZKYm8NxnW3l03iYqa+oY1COZrkmxpCZEs2jbfvL2V5AYE8m14/vwowsGHqyyO5qK6jpunL6AJdv3c9t5A1i18wDz1hUckWSbEhUhDOvVmUiBJduL+OmkQfzv2ScBbgaCv32wkfySSurqlYqaOjYXlNE1KYYbJvbllSV57DpQyV+uGsWUES1XG85etZvvPb+U3qnxTB3Zi7+8v56J/dJ44oaxJMZEkbe/gh1FFfRLT6RbcixVtfX85b31PPnxZlISYhBgr9fe5ht/bFQEkRFCdGQEA7oncebAdM4ckM5J3ZKIjYo47iRiScKYIKurV+rq9YhHxdbXK8vyipi/eR8LtuzlQEUNPzx/IGceZYqTHUUV/Pat1cz6cjcXDuvOPRcPbfR8EF/7y6r565wN/POLbW42X4VvnZbNL6YMOSLJ5RdXcs0TX7C3rJp7Lh7K5WN6UVxRyw3PLGDVjgNcOLwHs1ftRkT4xilZfP/ck0hJiDm4/7LcIp7+ZAufbSw8eAHr0SmOuy4azNSRGewoquAfn2/jhQXbKfa6PTe4YGh37pw8uFHVVn29snDrPl5YmMurS3fQKyWeX391GKf278ribfv5ZGMhJZU1DO3ZmaEZnfjT7HV8urGQv14zmktGZgCu6mnVzmKKK2ooqaqhrKqOyAg3pUxUhJAQE0VibCTJcdEM6p5MfEwkNXX1/GTGcl5ftpNvnZZNYWkVM5fvJC0xhpMzOxPpTUkzoV8a143vQ3xMJPvKqvn2PxaxeNt+Tu2fRkVNHcUVNXSOj2Zi/zRO7d+VLokxB0srryzJ4+TMFJ69aRypiTG8tnQHP5mxnC6JMZRV1TYajJqaEE1MVAR7iqu4dnxv7rpoCJ3iotlbWsWmgjL2FFdSUFJFYWkVVbX11NUrVbX1LMstYs2uQw/ojI4UkuOiOX9IN/7vipHN/jttiiUJY9qRCm9MSGtszC/l7x9sYHx2GtdN6NPsMetUG1UZlVTW8K1nF7J0exHXjO/N984ZQI/OTde/qyrr9pSwMb+Ucwd3O6KqqK5e2VJYxqqdB9iwp5TTB3TllH5pzca/aOs+7vrvl2zILyUmMoLqunqiIoS46EhKqw4lnAeuGMGVOUc8c8xvdfXK3a+t5PkF24mLjuCW0/tx61n9jmgz8lVZU8dv3lrNirwDdI6PJjkuit0HKlmed4A6n6n2UxKiOXdwN3596fBGJaN56/J57rOt9OmSwOCeneiVEs/mglLW7CphT0klt5zej9MHdPXrPPJLKvl0YyE7iyopqaylpLKGAd2SuOm0bP//UrAkYYxpQk1dPSWVtXRJjGl54yCprq3nuc+2sru4ktNOSmN8dhoJ0ZHk7i9n1c5i0hJjmNBCsvGHqjJ79R5GZHamZ+djb5QvraplwZa9lFTWMiIzpV3PYWZJwhhjTJOaSxJtv2+WMcaYkLEkYYwxpkmWJIwxxjTJkoQxxpgmWZIwxhjTJEsSxhhjmmRJwhhjTJMsSRhjjGlSWA2mE5ECYNtxHKIr0PSj0cJTRzxn6Jjn3RHPGTrmeft7zlmqeuSEYoRZkjheIrKoqVGH4aojnjN0zPPuiOcMHfO8A3nOVt1kjDGmSZYkjDHGNMmSRGNPhDqAEOiI5wwd87w74jlDxzzvgJ2ztUkYY4xpkpUkjDHGNMmShDHGmCZZkgBEZJKIrBORjSJyZ6jjCQYR6S0ic0VktYisEpEfeOu7iMh7IrLB+zM11LEGg4hEishSEXnTW84Wkfneb/6iiITu0WxBICIpIvKyiKwVkTUiMrEj/NYi8iPv3/dKEXleROLC8bcWkekiki8iK33WHfX3Fech7/xXiMgYf76rwycJEYkEHgYmA0OBa0VkaGijCopa4MeqOhQ4Bfiud553AnNUdQAwx1sORz8A1vgs/wH4i6qeBOwHbg5JVMHzV+AdVR0MjMSde1j/1iLSC7gNyFHV4UAkcA3h+Vs/C0w6bF1Tv+9kYID3mgY86s8XdfgkAYwHNqrqZlWtBl4ALg1xTAGnqrtUdYn3vgR30eiFO9fnvM2eA74amgiDR0QygSnAU96yAOcCL3ubhNV5i0hn4EzgaQBVrVbVIjrAbw1EAfEiEgUkALsIw99aVT8C9h22uqnf91LgH+p8AaSISM/WfpclCXehzPVZzvPWhS0R6QuMBuYD3VV1l/fRbqB7iMIKpgeBnwL13nIaUKSqtd5yuP3m2UAB8IxXxfaUiCQS5r+1qu4A/ghsxyWHA8Biwvu39tXU73tc1zhLEh2MiCQBrwA/VNVi38/U9YcOqz7RInIxkK+qi0MdywkUBYwBHlXV0UAZh1UthelvnYq7a84GMoBEjqyS6RAC+ftakoAdQG+f5UxvXdgRkWhcgvi3qv7XW72noejp/ZkfqviC5DRgqohsxVUlnourr0/xqiQg/H7zPCBPVed7yy/jkka4/9bnA1tUtUBVa4D/4n7/cP6tfTX1+x7XNc6SBCwEBng9IGJwDV0zQxxTwHn18E8Da1T1zz4fzQRu9N7fCLx+omMLJlW9S1UzVbUv7rf9QFW/DswFrvA2C6vzVtXdQK6IDPJWnQesJsx/a1w10ykikuD9e28477D9rQ/T1O87E7jB6+V0CnDAp1qqRTbiGhCRi3D11pHAdFX9bYhDCjgROR34GPiSQ3XzP8e1S7wE9MFNs36Vqh7eIBYWRORs4CeqerGI9MOVLLoAS4HrVbUqlPEFkoiMwjXUxwCbgW/ibgrD+rcWkfuBq3G9+ZYCt+Dq38PqtxaR54GzcVOC7wF+CbzGUX5fL2H+HVf1Vg58U1UXtfq7LEkYY4xpilU3GWOMaZIlCWOMMU2yJGGMMaZJliSMMcY0yZKEMcaYJlmSMKaNEJGzG2apNaatsCRhjDGmSZYkjPGTiFwvIgtEZJmIPO49q6JURP7iPctgjoike9uOEpEvvHn8X/WZ4/8kEXlfRJaLyBIR6e8dPsnnORD/9gZCGRMyliSM8YOIDMGN6D1NVUcBdcDXcZPJLVLVYcCHuBGwAP8AfqaqI3Cj3RvW/xt4WFVHAqfiZi0FNzvvD3HPNumHm3vImJCJankTY4yP84CxwELvJj8eN5FaPfCit82/gP96z3VIUdUPvfXPATNEJBnopaqvAqhqJYB3vAWqmuctLwP6Ap8E/7SMOTpLEsb4R4DnVPWuRitF7jlsu2Od78Z3TqE67P+oCTGrbjLGP3OAK0SkGxx8rnAW7v9Sw0yj1wGfqOoBYL+InOGt/wbwofdkwDwR+ap3jFgRSTihZ2FMK9ldijF+UNXVInI3MFtEIoAa4Lu4B/uM9z7Lx7VbgJuy+TEvCTTMxgouYTwuIr/yjnHlCTwNY1rNZoE1JgBEpFRVk0IdhzGBZtVNxhhjmmQlCWOMMU2ykoQxxpgmWZIwxhjTJEsSxhhjmmRJwhhjTJMsSRhjjGnS/weQIOP/YyomngAAAABJRU5ErkJggg==)

Pada hasil proses evaluation, proses training model cukup smooth dan model konvergen pada epochs sekitar 100. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.1862 dan error pada data validasi sebesar 0.2065. Nilai tersebut cukup bagus untuk sistem rekomendasi.

