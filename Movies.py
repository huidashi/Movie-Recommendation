#!/usr/bin/env python
# coding: utf-8

# # The Movies Dataset

# 
# 
# dataset : https://www.kaggle.com/rounakbanik/the-movies-dataset/data  
# cleaned : https://drive.google.com/drive/folders/1ZGp7ORu9nA6l3PyNK_H0MGTXNl4sNMsA?usp=sharing

# In[1]:


import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import wordcloud, STOPWORDS 
import glob
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import functools
import operator


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


movies = pd.read_csv('/Users/davidshi/Downloads/movie/movies.csv', header=0)
movies = movies.replace({np.nan: None}) # replace NaN with None
movies.head()


# In[4]:


movies.isnull().sum()


# In[5]:


movies.shape


# In[6]:


def get_year(date):
    year = None
    if date:
        year = date[:4]
    return year

movies['year'] = movies.date.apply(get_year)


#  ## Quick Explanatory
#  
# Which country have twhe most movies produced?

# In[7]:


r = movies[~movies['production_countries'].isnull()]


# In[8]:


r =list(r['production_countries'].str.split(', '))


# In[9]:


flattened_list = functools.reduce(operator.iconcat, r, [])


# In[10]:


dd= pd.Series(flattened_list)


# In[11]:


dd.value_counts().head(10).plot(kind='bar',figsize=(10, 5))
plt.title('Production Countries Movie Counts')


# ## Multilabel Classification on Genre
# 
# #### Using keywords column to predict Genre
# 
# Each movie is classified as mulitple genres. I am going to predict the muliple genres using keywords column. String are cleaned and vectorized.

# In[12]:


df = movies.dropna(subset = ['title','genres','description'])


# In[13]:


mlb = MultiLabelBinarizer()
mlb.fit(df['genres'])

y = mlb.transform(df['genres'])


# In[14]:


def cleanstring(x):
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = [c for c in x.split() if c.lower() not in stopwords.words('english')]
    return " ".join(x)


# In[15]:


def removepunc(x):
    
    x = x.translate(str.maketrans('', '', string.punctuation))
    return x


# In[16]:


df['keywords'] = df['keywords'].astype(str)


# In[17]:


X=df['keywords'].apply(removepunc)
X=TfidfVectorizer('english',max_df=0.8,lowercase=True).fit_transform(X)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=0)


# In[18]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
results = pd.DataFrame(columns = ['F1 Score']) 


# In[19]:


for i in range (10):
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=i)
    clf.fit(Xtrain,ytrain)
    pred=clf.predict(Xtest)
    results=results.append({'F1 Score' : f1_score(ytest,pred,average="micro")}, ignore_index=True)


# In[20]:


plt.figure(figsize=(8,8))
plt.title('F1 Score')
sns.boxplot(x=results['F1 Score'])


# #### Using description column to predict Genre

# In[21]:


X2 = df['description'].astype(str)
X2 = TfidfVectorizer('english',max_df=0.8,lowercase=True).fit_transform(X2)


# In[22]:


results = pd.DataFrame(columns = ['F1 Score']) 
for i in range (10):
    Xtrain,Xtest,ytrain,ytest = train_test_split(X2,y,test_size=0.2,random_state=i)
    clf.fit(Xtrain,ytrain)
    pred=clf.predict(Xtest)
    results=results.append({'F1 Score' : f1_score(ytest,pred,average="micro")}, ignore_index=True)
plt.figure(figsize=(8,8))
plt.title('F1 Score')
sns.boxplot(x=results['F1 Score'])


# ## movie recommendations by rating and popularity

# Although data already provides average rating, this can be quite misleading. When comparing two average ratings the consideration of number of votes is very important. Movie 1 can have a rating of 9.0 with only 3 votes compared to Movie 2 having a rating of 8.0 with 50 votes. In this situation Movie 1's rating is very biased due to the low amount of voters. To tackle this problem IMBD have a weight rating formula.
# 
# Weighted Rating = (vR)/(v+m) + (mC)/(v+m)
# 
# m = minimum votes to be listed in chart ( we will assume movies with 85 percentile number of voters to be relevant)  
# C = mean of average rating for all movies  
# R = average rating for specific movie  
# v = number of votes for specific movie

# In[23]:


m=df['num_votes'].quantile(0.85)
C=df['average_vote'].mean()
df2=df[df['num_votes']>m]


# In[24]:


def weightedrating(j,m=m,C=C):
    v=j['num_votes']
    R=j['average_vote']
    
    return (v*R)/(v+m) + (m*C)/(v+m)


# In[25]:


df2['imbd_wr']=df.apply(weightedrating,axis=1)


# In[26]:


df.shape


# In[27]:


df2.shape


# In[28]:


df.set_index('id')
df2.set_index('id')
df['imbd_wr']=df2['imbd_wr']


# In[29]:


df['genres']=df['genres'].str.split(', ')
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_df = df.drop('genres', axis=1).join(s)


# This seperates the genre column by listing movies duplicately in dataframe. Example would be:  
# 
# --before--  
# movie 1 : [comedy,dance]        
# 
# --after--   
# movie 1 : [comedy]  
# movie 1 : [dance]

# In[30]:


def genre_rec(genre,method):
    if(method == 'popularity'):
        x = gen_df[gen_df['genre']==genre]
        x=x.reset_index(drop=True)
        return x[['title','year',method]].sort_values(by=[method],ascending=False).head(20)
    
    if(method =='imbd_wr'):
        x = gen_df[gen_df['genre']==genre]
        x=x.reset_index(drop=True)
        return x[['title','year',method]].sort_values(by=[method],ascending=False).head(20)


# In[31]:


genre_rec('horror','imbd_wr')


# In[32]:


genre_rec('comedy','popularity')


# In[33]:


plt.figure(figsize=(20,6))
plt.ylim(5,6.5)
sns.barplot(x='genre',y='average_vote',data=gen_df)
plt.xticks(rotation=45)
plt.title('Genre Average Ratings', fontsize=20)


# ## movie recommendations by plot similarity
# 
# 

# In[34]:


tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(df['description'])


# In[35]:


tfidf_matrix.shape


# description column is vectorized and transformed into TF-IDF matrix. 70000+ words describing 40000+ movies.

# In[36]:


df = df.reset_index(drop=True)


# In[37]:


indices = pd.Series(df.index, index=df['title'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[38]:


indices ##movie array


# redoing index so index can match between movie array and cosine similiarty matrix. Calculating the dot product / scalar product of TF-IDF vectorized data will give cosine similiarity score.

# In[39]:


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of movie 
    idx = indices[title]

    # pair similarity scores 
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

   
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


# In[40]:


get_recommendations('Minions')


# In[41]:


get_recommendations('Toy Story')


# movie recommendation by cast, keywords, genre. Using count vectorizer to weigh frequency of repeating cast and repeating keywords in multiple movies.

# In[42]:


df.isnull().sum()


# In[43]:


df = df.dropna(subset = ['cast'])
df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['title'])


# In[44]:


df['cast'].apply(removepunc)


# In[45]:


df['keywords'].apply(removepunc)


# In[46]:


def listToString(s): 
    str1 = " " 
    return (str1.join(s))


# In[47]:


df['genres'].apply(listToString)


# In[48]:


X = df['cast'].apply(removepunc) + df['keywords'].apply(removepunc) + df['genres'].apply(listToString)

count_matrix = CountVectorizer('english').fit_transform(X)

cosine_sim2 = cosine_similarity(count_matrix,count_matrix)


# In[49]:


get_recommendations('Minions',cosine_sim2)


# In[50]:


get_recommendations('Toy Story', cosine_sim2)


# Using different columns to get recommendations did yield different results

# ### movie recommendations by user history & plot similarity
# 
# Used user rating csv to find top movies of specific users and recommended movies based on that.

# In[141]:


user = pd.read_csv('/Users/davidshi/Downloads/movie/ratings_small.csv')


# In[142]:


user


# User data contains rows of user ratings. rating column (1-5). 

# In[143]:


links = pd.read_csv('/Users/davidshi/Downloads/movie/links.csv')


# In[144]:


links


# links df have movieId and tmbdid equivalent. Need links df to get movie title for user df  

# In[145]:


user = user.merge(links[['movieId','tmdbId']],on='movieId',how='left')


# In[146]:


user = user.rename(columns={'tmdbId':'id'})
user = user.drop(columns =['movieId'])


# In[147]:


user


# In[148]:


user = user.merge(movies[['id','title','average_vote']],on='id',how='left')


# In[149]:


user


# In[150]:


user.isnull().sum()


# In[151]:


user = user.dropna(subset = ['id','title'])


# In[152]:


user


# In[153]:


user.userId.value_counts().min()


# Users rated atleast 19 movies

# In[154]:


user = user.sort_values(by=['userId','rating','average_vote'],ascending=[True,False,False]).groupby('userId').head(5)


# In[155]:


#get top 5 movies for user

def getTopMov(uid) :
    return user[user['userId']==uid].title.values
    
    


# In[156]:


#recommendation by finding 10 highest pair similartiy from list of movies

def get_recommendations2(title,cosine_sim):
    sim_scores=[]
    for i in title:
        # Get the index of movie 
        idx = indices[i]

        # pair similarity scores 
        sim_scores = sim_scores + list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

   
    sim_scores = sim_scores[5:15]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


# In[157]:


getTopMov(10)


# In[158]:


get_recommendations2(getTopMov(10), cosine_sim)


# In[159]:


#recommendation by finding 3 highest pair similartiy for each movie in list

def get_recommendations3(title,cosine_sim):
    sim_scores_final =[]
    for i in title:
        # Get the index of movie 
        idx = indices[i]

        # pair similarity scores 
        sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores_final = sim_scores_final + sim_scores[1:4]
        
    
    movie_indices = [i[0] for i in sim_scores_final]
    
    return df['title'].iloc[movie_indices]


# In[160]:


getTopMov(10)


# In[161]:


get_recommendations3(getTopMov(10),cosine_sim)


# In[ ]:





# In[ ]:





# In[ ]:




