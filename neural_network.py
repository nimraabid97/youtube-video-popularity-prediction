import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score


# Load and preprocess the dataset
preprocessed_data = pd.read_csv('Dataset/test.csv')

# Handle NaN values in target variables
imputer = SimpleImputer(strategy='median')
preprocessed_data['score'] = imputer.fit_transform(preprocessed_data[['score']])


# Identify non-numeric features and encode or drop them
non_numeric_columns = preprocessed_data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in non_numeric_columns:
    preprocessed_data[column] = label_encoder.fit_transform(preprocessed_data[column])

# Splitting the dataset
X = preprocessed_data.drop(['views', 'likes', 'comments_count', 'score'], axis=1)
y = preprocessed_data[['score']]

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBRegressor models
xgb_model = xgb.XGBClassifier(random_state=42)
#model_score_xgb = XGBRegressor(objective='reg:squarederror')

xgb_model.fit(X_train,y_train)
#model_score_xgb.fit(X_train, y_train['score'])

# Make predictions
y_pred_score_xgb =xgb_model.predict(X_test)
#y_pred_score_xgb = model_score_xgb.predict(X_test)

# Calculate RMSE for each target
#rmse_score_xgb = mean_squared_error(y_test['score'], y_pred_score_xgb, squared=False)

results = confusion_matrix(y_test,y_pred_score_xgb)
print(results)

accuracy = accuracy_score(y_test,y_pred_score_xgb) * 100
print(accuracy)


#print(f'RMSE for Score: {rmse_score_xgb}')
