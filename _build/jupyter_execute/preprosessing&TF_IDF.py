#!/usr/bin/env python
# coding: utf-8

# #Preprosessing data & TF - IDF

# ## Import Library
# 
# ### Library yang digunakan
# 
# - **Pandas**
# 
# - **swifter**
# 
# - **PySastrawi**
# 
# - **scikit-learn**
# 
# - **networkx**
# 
# - **nltk**
# 
# - **numpy**
# 
# - **re**

# In[1]:


import pandas as pd
import numpy as np
import string
import re #regrex libray
import nltk
import swifter
import Sastrawi
import networkx as nx

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# ## Preprosessing Data

# Ambil data abstrak yang sudah di ektraksi dari pta trunojoyo tadi.

# In[2]:


df = pd.read_csv('hasil-crawl.csv')
df.head()


# Kemudian cek apakah ada missing value pada data yang di crawl.

# In[3]:


df.isna().sum()


# Jika ada missing value, hilangkan missing value nya dengan fungsi dropna.

# In[4]:


df = df.dropna(axis=0, how='any')
df.isna().sum()


# 

# # Case Folding

# Lalu lakukan case folding (mengganti semua data menjadi lowercase)terhadap data df.

# In[5]:


# ------ Case Folding --------
# gunakan fungsi Series.str.lower() pada Pandas
df['abstrak'] = df['abstrak'].str.lower()


print('Case Folding Result : \n')
print(df['abstrak'].head(20))
print('\n\n\n')


# # Tokenizing

# Kemudian hilangkan karakter yang tidak termasuk datam ASCII, hilangkan angka, hilangkan simbol - simbol, dan juga spasi kosong.
# 
# Kemudian lakukan tokenizing pada data df tersbut.

# In[6]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# ------ Tokenizing ---------

def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
df['abstrak'] = df['abstrak'].apply(remove_tweet_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['abstrak'] = df['abstrak'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['abstrak'] = df['abstrak'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

df['abstrak'] = df['abstrak'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

df['abstrak'] = df['abstrak'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

df['abstrak'] = df['abstrak'].apply(remove_singl_char)

# NLTK word rokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['abstrak_tokens'] = df['abstrak'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(df['abstrak_tokens'].head(20))
print('\n\n\n')


# Setelah itu hitung frekuensi token yang sering muncul didalam dokumen dengan NTLK
# 

# In[7]:


# NLTK calc frequency distribution
def freqDist_wrapper(text):
    return FreqDist(text)

df['abstrak_tokens_fdist'] = df['abstrak_tokens'].apply(freqDist_wrapper)

print('Frequency Tokens : \n') 
print(df['abstrak_tokens_fdist'].head(20).apply(lambda x : x.most_common()))
ab = df['abstrak_tokens_fdist'].head(20).apply(lambda x : x.most_common())


# # Filtering (Stopword Removal)

# Setelah itu Lakukan filtring stopword dengan library ntlk

# In[8]:


from nltk.corpus import stopwords

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("hasil-crawl.csv", names= ["stopwords"], header = None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

df['abstrak_tokens_WSW'] = df['abstrak_tokens'].apply(stopwords_removal) 


print(df['abstrak_tokens_WSW'].head(20))


# # Normalization

# Kemudian Normalisasikan dokumen menggunakan fungsi normalized_term yang dibuat dengan bantuan Library Dictionary

# In[9]:


normalizad_word = pd.read_csv("hasil-crawl.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

df['abstrak_normalized'] = df['abstrak_tokens_WSW'].apply(normalized_term)

df['abstrak_normalized'].head(20)


# # Stemming

# Setelah itu lakukan Stemming data mengguanakan Library sastrawi

# In[10]:


# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in df['abstrak_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df['abstrak_tokens_stemmed'] = df['abstrak_normalized'].swifter.apply(get_stemmed_term)
print(df['abstrak_tokens_stemmed'])


# Kemudian Rubah hasil data menjadi csv dan melanjutkannya ke proses pengolahan data yang selanjutnya

# In[11]:


#df.to_csv("Hasil_Text_Preprocessing.csv")


# # Prepare Data

# Masukan data yang telah dipreprosessing tadi.

# In[12]:


import pandas as pd 
import numpy as np

df = pd.read_csv("Hasil_Text_Preprocessing.csv", usecols=["abstrak_tokens_stemmed"])
df.columns = ["abstrak"]

df.head(20)


# # Term Frekuensi

# Hitung Frekuensi Term yang muncul dalam 1 dokumen ini dengan counVectorizer.

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

a=len(document)
document = df['abstrak']

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())


# In[14]:


a = vectorizer.get_feature_names()


# # TF-IDF

# ## Ekstraksi fitur dan membuat Document Term Matrix (DTM)
# 
# Dalam perhitungan LSA (Latent Semantic Analysis) data yang diperlukan hanya TF-IDF. Sehingga pada program ini tidak perlu mencari nilai TF dari dokumen. Untuk mengetahui nilai TF-IDF dapat dilakukan dengan membuat objek dari kelas TfidfVectorizer yang disediakan library scikit-learn.
# 
# Rumus Term Frequency (TF):
# 
# $$
# tf(t,d) = { f_{ t,d } \over \sum_{t' \in d } f_{t,d}}
# $$
# 
# $ f_{ t,d } \quad\quad\quad\quad$: Jumlah kata t muncul dalam dokumen
# 
# $ \sum_{t' \in d } f_{t,d} \quad\quad$: Jumlah seluruh kata yang ada dalam dokumen
# 
# Rumus Inverse Document Frequency (IDF):
# 
# $$
# idf( t,D ) = log { N \over { | \{ d \in D:t \in d \} | } }
# $$
# 
# $ N \quad\quad\quad\quad\quad$ : Jumlah seluruh dokumen
# 
# $ | \{ d \in D:t \in d \} | $ : Jumlah dokumen yang mengandung kata $ t $
# 
# Rumus TF - IDF:
# 
# $$
# tfidf( t,d,D ) = tf( t,d ) \times idf( t,D )
# $$

# rubah bentuk tf - idf nya menjadi array agar bisa dimasukan kedalam dataframe.

# In[15]:


tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(document)).toarray()


# lalu masukkan array tadi menjadi sebuah tabel dengan bantuan pandas dataframe.

# In[16]:


dfb = pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb


# Kemudian convert hasil TF-IDF tadi menjadi csv untuk diproses lebih lanjut.

# In[17]:


#dfb.to_csv("Hasil_TF-IDF.csv")

