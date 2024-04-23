import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset
preprocessed_data = pd.read_csv('Dataset/preprocessed.csv')

preprocessed_data = preprocessed_data.dropna(subset=['comments_count'])

# Identify non-numeric columns
non_numeric_columns = preprocessed_data.select_dtypes(include=['object']).columns

# Encode non-numeric columns
label_encoder = LabelEncoder()
for column in non_numeric_columns:
    preprocessed_data[column] = label_encoder.fit_transform(preprocessed_data[column])

# Selecting features and target variables
X = preprocessed_data.drop(['views', 'likes', 'comments_count'], axis=1)
y = preprocessed_data[['views', 'likes', 'comments_count']]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor models
model_views_rf = RandomForestRegressor(random_state=42)
model_likes_rf = RandomForestRegressor(random_state=42)
model_comments_rf = RandomForestRegressor(random_state=42)

model_views_rf.fit(X_train, y_train['views'])
model_likes_rf.fit(X_train, y_train['likes'])
model_comments_rf.fit(X_train, y_train['comments_count'])

# Make predictions and calculate RMSE
y_pred_views_rf = model_views_rf.predict(X_test)
y_pred_likes_rf = model_likes_rf.predict(X_test)
y_pred_comments_rf = model_comments_rf.predict(X_test)

rmse_views_rf = mean_squared_error(y_test['views'], y_pred_views_rf, squared=False)
rmse_likes_rf = mean_squared_error(y_test['likes'], y_pred_likes_rf, squared=False)
rmse_comments_rf = mean_squared_error(y_test['comments_count'], y_pred_comments_rf, squared=False)

print(f'RMSE for Views: {rmse_views_rf}')
print(f'RMSE for Likes: {rmse_likes_rf}')
print(f'RMSE for Comments Count: {rmse_comments_rf}')
