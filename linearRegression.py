#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# Load and preprocess the dataset
preprocessed_data = pd.read_csv('Dataset/preprocessed.csv')



# Handle NaN values in target variables
imputer = SimpleImputer(strategy='median')
preprocessed_data['views'] = imputer.fit_transform(preprocessed_data[['views']])
preprocessed_data['likes'] = imputer.fit_transform(preprocessed_data[['likes']])
preprocessed_data['comments_count'] = imputer.fit_transform(preprocessed_data[['comments_count']])

# Convert target variables to categories and encode them
preprocessed_data['views'] = pd.qcut(preprocessed_data['views'], q=4, duplicates='drop').astype(str)
preprocessed_data['likes'] = pd.qcut(preprocessed_data['likes'], q=4, duplicates='drop').astype(str)
preprocessed_data['comments_count'] = pd.qcut(preprocessed_data['comments_count'], q=4, duplicates='drop').astype(str)
label_encoder = LabelEncoder()
preprocessed_data['views'] = label_encoder.fit_transform(preprocessed_data['views'])
preprocessed_data['likes'] = label_encoder.fit_transform(preprocessed_data['likes'])
preprocessed_data['comments_count'] = label_encoder.fit_transform(preprocessed_data['comments_count'])

# Identify non-numeric features and encode or drop them
non_numeric_columns = preprocessed_data.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    # Assuming these are categorical, encode them. If not, consider dropping them.
    preprocessed_data[column] = label_encoder.fit_transform(preprocessed_data[column])

# Splitting the dataset
X = preprocessed_data.drop(['views', 'likes', 'comments_count'], axis=1)
y = preprocessed_data[['views', 'likes', 'comments_count']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = linear_model.LinearRegression()
model.fit(X_train, y_train['views'])


YouTube_Views_predicted = model.predict(X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, YouTube_Views_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
#
# plt.show()

# Mean squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698
