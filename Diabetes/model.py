import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Define the features and target variable
X = df.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = df['Outcome']  # Target variable ('Outcome' column)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model (optional, can be printed out to see how good it is)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file (diabetes_model.pkl)
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model has been saved to 'diabetes_model.pkl'")
