import pandas as pd

file_path = 'Dataset/commentsPreprocessed.csv'
data = pd.read_csv(file_path, encoding='unicode_escape')
positiveVideos = []
negativeVideos = []
neutralVideos = []
for i in range(0, data.video_id.nunique()):
    positive = data[(data.video_id == data.video_id.unique()[i]) & (data.Sentiment == 2)].count()[0]
    negative = data[(data.video_id == data.video_id.unique()[i]) & (data.Sentiment == 0)].count()[0]
    neutral = data[(data.video_id == data.video_id.unique()[i]) & (data.Sentiment == 1)].count()[0]

    total = data[data.video_id == data.video_id.unique()[i]]['Sentiment'].value_counts().sum()
    positivePercentage = (positive / total) * 100
    negativePercentage = (negative / total) * 100
    neutralPercentage = (neutral / total) * 100
    positiveVideos.append(round(positivePercentage, 2))
    negativeVideos.append(round(negativePercentage, 2))
    neutralVideos.append(round(negativePercentage, 2))
processed_data = {
    'positive_videos': positiveVideos,
    'negative_videos': negativeVideos,
    'neutral_videos': neutralVideos,

}
sentimentScores = pd.DataFrame(processed_data, data.video_id.unique())
sentimentScores.to_csv("Dataset/sentimentScores.csv")
