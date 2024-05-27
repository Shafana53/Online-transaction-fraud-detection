import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn import preprocessing

# Load the dataset
dataset_path = 'transactions_data.csv'
df = pd.read_csv(dataset_path)

# Drop unnecessary columns (adjust as needed)
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Convert categorical features to numerical using label encoding
label_encoder = preprocessing.LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Split the data into features (X) and target (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Standard Scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the resampled data
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

print("Random Forest Classifier Results:")
print(f'Accuracy: {accuracy_rf:.2f}')
print('Classification Report:\n', classification_report_rf)

# Save the trained Random Forest model
model_save_path_rf = 'fraud_model_rf.pkl'
joblib.dump(rf_model, model_save_path_rf)
