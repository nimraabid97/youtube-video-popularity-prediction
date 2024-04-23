# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
import re
from Utilities import preProcessingHelper
from imblearn.over_sampling import SMOTE

file_path = '../Dataset/pakComments.csv'
data1 = pd.read_csv(file_path, encoding='unicode_escape')



data1["comment_text"] = data1["comment_text"].fillna("No Comments")

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data1["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data1["comment_text"]]
data1["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data1["comment_text"]]
data1["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data1["comment_text"]]
data1['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data1["comment_text"]]
score = data1["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05:
        sentiment.append('Positive')
    elif i <= -0.05:
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data1["Sentiment"] = sentiment
data2 = data1.drop(['Positive', 'Negative', 'Neutral', 'Compound'], axis=1)
print(data2.head());

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4*')
data_copy1 = data2.copy()
data_copy1['comment_text'] = data_copy1['comment_text'].apply(preProcessingHelper.text_processing)

le = LabelEncoder()
data_copy1['Sentiment'] = le.fit_transform(data_copy1['Sentiment'])

processed_data = {
    'video_id': data_copy1.video_id,
    'Comment': data_copy1.comment_text,
    'Sentiment': data_copy1['Sentiment']
}

processed_data = pd.DataFrame(processed_data)

processed_data['Sentiment'].value_counts()
processed_data.to_csv("../Dataset/commentsPreprocessed.csv")

# Balancing data

df_neutral = processed_data[(processed_data['Sentiment'] == 1)]
df_negative = processed_data[(processed_data['Sentiment'] == 0)]
df_positive = processed_data[(processed_data['Sentiment'] == 2)]
print(df_neutral.shape)
print(df_negative.shape)
print(df_positive.shape)
# upsample minority classes
# df_negative_upsampled = resample(df_negative,
#                                  replace=True,
#                                  n_samples=45535,
#                                  random_state=42)
# #
# df_neutral_upsampled = resample(df_neutral,
#                                 replace=True,
#                                 n_samples=45535,
#                                 random_state=42)
#
# # Concatenate the upsampled dataframes with the neutral dataframe
# data_copy = pd.concat([df_negative_upsampled, df_neutral_upsampled, df_positive])
# #
# data_copy['Sentiment'].value_counts()
# data_copy.to_csv("../Dataset/commentsPreprocessed.csv")
# # final_data.to_csv("commentsPreprocessed.csv")

