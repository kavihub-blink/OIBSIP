import pandas as pd
#load dataset
df=pd.read_csv("../dataset/spam.csv",encoding="latin-1")
#keep only necessary columns
df=df[['v1','v2']]
#preview data
print(df.head())

#convert labels to numeric
df['label']=df['v1'].map({'ham':0, 'spam':1})
#features and target
X = df['v2']
y = df['label']

#convert text to features
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()
X_features=cv.fit_transform(X)
#split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_features, y, test_size=0.2, random_state=42)
#train Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train, y_train)

#test accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred=model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))

#test New email
new_email=["you won a free lottery! Claim Now!"]
new_feature=cv.transform(new_email)
prediction=model.predict(new_feature)
if prediction[0]==1:
    print("Spam")
else:
    print("Ham")
    
#Save Model
import joblib
joblib.dump(model, "spam_model.pkl")
joblib.dump(cv, "vectorizer.pkl")
