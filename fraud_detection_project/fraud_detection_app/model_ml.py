import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing

import joblib


# Load the dataset
# transactions = Transaction.objects.all()
# df = pd.DataFrame.from_records(Transaction.objects.values())
dataset_path = r'C:\Users\AaRu\fraud_detection_project\fraud_detection_app\transaction_data.csv.csv'
df = pd.read_csv(r'C:\Users\AaRu\fraud_detection_project\fraud_detection_app\transaction_data.csv.csv')

# Drop unnecessary columns (adjust as needed)
df = df.drop(['nameOrig', 'nameDest', 'step'], axis=1)

# Convert categorical features to numerical using label encoding
label_encoder = preprocessing.LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])
print(df.head(10))

# Split the data into features (X) and target (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize the Decision Tree Classifier with feature names
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report_result)


# Save the trained model
model_save_path = r'A:\Minor\fraud_model.pkl'
joblib.dump(dt_model, model_save_path)