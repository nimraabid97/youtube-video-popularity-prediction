import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

file_path = 'Dataset/pakComments.csv'
Pak_comments = pd.read_csv(file_path)

# Pak_comments.head()
#

Pak_comments['comment_text'] = Pak_comments['comment_text'].fillna('No Comments')
Pak_comments['likes'] = Pak_comments['likes'].fillna(0)
Pak_comments['replies'] = Pak_comments['replies'].fillna(0)
Pak_comments.drop(41587, inplace=True)
#
Pak_comments = Pak_comments.reset_index().drop('index', axis=1)
#
Pak_comments.likes = Pak_comments.likes.astype(int)
Pak_comments.replies = Pak_comments.replies.astype(int)
#
# # Removing Punctuations, Numbers and Special Characters.
Pak_comments['comment_text'] = Pak_comments['comment_text'].str.replace("[^a-zA-Z#]", " ")
#
# # Removing Short Words.
#
Pak_comments['comment_text'] = Pak_comments['comment_text'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
#
# # Changing the text to lower case.
#
Pak_comments['comment_text'] = Pak_comments['comment_text'].apply(lambda x: x.lower())
#
# # Tokenization
tokenized_tweet = Pak_comments['comment_text'].apply(lambda x: x.split())
# tokenized_tweet.head()
#
# Lemmatization
#
wnl = WordNetLemmatizer()
tokenized_tweet.apply(lambda x: [wnl.lemmatize(i) for i in x if i not in set(stopwords.words('english'))])
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

Pak_comments['comment_text'] = tokenized_tweet

Pak_comments.to_csv("preprocessedComments.csv")
#
# # Sentiment Analysis on the Comments Dataset
#
# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()
#
# Pak_comments['Sentiment Scores'] = Pak_comments['comment_text'].apply(lambda x:sia.polarity_scores(x)['compound'])
# # Classifying the Sentiment scores as Positive, Negative and Neutra
# Pak_comments['Sentiment'] = Pak_comments['Sentiment Scores'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))
#
# Pak_comments.head();
# Pak_comments.Sentiment.value_counts()
#
# #calculate the percentage of comments which are positive in all the videos
