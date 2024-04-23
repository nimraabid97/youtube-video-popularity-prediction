import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score


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

# Initialize and train the XGB models
xgb_model = xgb.XGBClassifier(random_state=42)

xgb_model.fit(X_train,y_train)

# Make predictions
y_pred_score_xgb =xgb_model.predict(X_test)

results = confusion_matrix(y_test,y_pred_score_xgb)
print(results)

accuracy = accuracy_score(y_test,y_pred_score_xgb) * 100
print('Accuracy (xgBoost):', accuracy)

f1_score = f1_score(y_test, y_pred_score_xgb, average='weighted')
print('f1_score (xgBoost):', f1_score)

#p
