#Import Libraries
import pandas as pd
import pickle
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer




#Load the corpus
corpus = pickle.load(open("corpus.pkl", "rb"))

#Load the model
model = pickle.load(open("Spam_sms_prediction.pkl", "rb"))

#Load the test data
test_data = pd.read_csv('Spam SMS Collection', sep='\t', names=['label','message'])
test_data.drop_duplicates(inplace=True)
test_data.reset_index(drop=True, inplace=True)

#Cleaning the test messages
ps = PorterStemmer()
test_corpus = []
for i in range(0, test_data.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=test_data.message[i])
    message = message.lower()
    words = message.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    message = ' '.join(words)
    test_corpus.append(message)

"""#Creating the Bag of Words model for the test data
cv = CountVectorizer(max_features=2500)
X_test = cv.fit_transform(test_corpus).toarray() # type: ignore"""

tfidf = TfidfVectorizer(max_features=2500)
X_test = tfidf.fit_transform(corpus).toarray()

#Extracting dependent variable from the test dataset
y_test = pd.get_dummies(test_data['label'])
y_test = y_test.iloc[:, 1].values

#Predicting the Test set results
y_pred = model.predict(X_test)

#Calculating the Accuracy
accuracy = accuracy_score(y_test, y_pred)

target_names = ['ham', 'spam']

print("Accuracy:", accuracy)

# Generating classification report
print(classification_report(y_test, y_pred, target_names=target_names))