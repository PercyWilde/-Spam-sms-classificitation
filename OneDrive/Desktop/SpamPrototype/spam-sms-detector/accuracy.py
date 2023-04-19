import pickle
from sklearn.metrics import accuracy_score

# Load the saved model
model_file = open('Spam_sms_prediction.pkl', 'rb')
model = pickle.load(model_file)
model_file.close()

# Load the test set
X_test = ... # load the test data
y_test = ... # load the test labels

# Make predictions using the loaded model
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
