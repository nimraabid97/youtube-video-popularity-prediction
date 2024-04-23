import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import isodate

# Load the dataset
youTube_videos_path = '../Dataset/pakVideos.csv'
comments_path = '../Dataset/sentimentScores.csv'

youtube_data = pd.read_csv(youTube_videos_path)
comments_data = pd.read_csv(comments_path)

# Preprocessing steps
#Merge Data
youtube_data = youtube_data.merge(comments_data, how='left', on='video_id')

# 1. Convert 'publish_time' to datetime and extract features
youtube_data['publish_time'] = pd.to_datetime(youtube_data['publish_time'])
youtube_data['year'] = youtube_data['publish_time'].dt.year
youtube_data['month'] = youtube_data['publish_time'].dt.month
youtube_data['day'] = youtube_data['publish_time'].dt.day
youtube_data['hour'] = youtube_data['publish_time'].dt.hour

# 2. Handle missing values
# For simplicity, fill missing values in 'likes' with the median (other strategies can be used)

youtube_data['likes'] = youtube_data['likes'].fillna(0)
youtube_data['positive_videos'] = youtube_data['positive_videos'].fillna(0)
youtube_data['negative_videos'] = youtube_data['negative_videos'].fillna(0)
youtube_data['neutral_videos'] = youtube_data['neutral_videos'].fillna(0)

# Fill missing categorical data with a placeholder value
youtube_data['tags'] = youtube_data['tags'].fillna('No Tags')
youtube_data['description'] = youtube_data['description'].fillna('No Description')

youtube_data['description'] = youtube_data['description'].str.replace('\d+', '')


# 3. Convert duration to seconds
def convert_duration(duration_str):
    # Parse a duration string from YouTube into seconds
    try:
        duration = isodate.parse_duration(duration_str)
        return duration.total_seconds()
    except ValueError:
        return 0


youtube_data['duration_seconds'] = youtube_data['duration'].apply(convert_duration)

# 4. Encode categorical data
# Using Label Encoding as an example (One-Hot Encoding or others could be considered)
encoder = LabelEncoder()
youtube_data['category_encoded'] = encoder.fit_transform(youtube_data['category'])
youtube_data['channel_title_encoded'] = encoder.fit_transform(youtube_data['channel_title'])

# 5. Normalization/Scaling of numerical features
scaler = MinMaxScaler()
numerical_features = ['duration_seconds', 'year', 'month', 'day', 'hour']
youtube_data[numerical_features] = scaler.fit_transform(youtube_data[numerical_features])


# Display the preprocessed dataset
youtube_data.to_csv("../Dataset/finalVideos.csv")

