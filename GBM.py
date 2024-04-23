import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import re

# Load and preprocess the dataset
preprocessed_data = pd.read_csv('Dataset/test.csv')
preprocessed_data = preprocessed_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
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

# Create the model
model = LGBMClassifier(learning_rate=0.1, n_estimators=100, num_leaves=31)

# Train the model
model.fit(X_train, y_train)

# Make class predictions
y_pred = model.predict(X_test)

# Make probability predictions
y_prob = model.predict_proba(X_test)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

f1_score = f1_score(y_test, y_pred, average='weighted')
print('f1_score', f1_score)
