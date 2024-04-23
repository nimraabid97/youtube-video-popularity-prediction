import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load and preprocess the dataset
preprocessed_data = pd.read_csv('Dataset/test.csv')


# Handle NaN values in target variables
imputer = SimpleImputer(strategy='median')
preprocessed_data['score'] = imputer.fit_transform(preprocessed_data[['score']])


# Convert target variables to categories and encode them

label_encoder = LabelEncoder()
preprocessed_data['score'] = label_encoder.fit_transform(preprocessed_data['score'])


# Identify non-numeric features and encode or drop them
non_numeric_columns = preprocessed_data.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    # Assuming these are categorical, encode them. If not, consider dropping them.
    preprocessed_data[column] = label_encoder.fit_transform(preprocessed_data[column])

# Splitting the dataset
X = preprocessed_data.drop(['views', 'likes', 'comments_count', 'score'], axis=1)
y = preprocessed_data[['score']]

# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier models
model_score_rf = RandomForestClassifier(random_state=42)
model_score_rf.fit(X_train, y_train['score'])

# Make predictions
y_pred_score_rf = model_score_rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test,y_pred_score_rf) * 100
print(f'accuracy Score for Random Forest: {accuracy}')

# Calculate F1 score
f1_score_rf = f1_score(y_test['score'], y_pred_score_rf, average='weighted')
print(f'F1 Score for Random Forest: {f1_score_rf}')
