import pandas as pd

from pandas_profiling import ProfileReport
df = pd.read_csv('Dataset/test.csv')

profile = ProfileReport(df)
profile.to_file(output_file = "youTube_videos.html")
