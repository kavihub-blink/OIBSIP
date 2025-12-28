from sklearn.datasets import load_iris
iris = load_iris()
X=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
#model train
model.fit(X_train,y_train)
print("Model training completed")
#make predictions
y_pred=model.predict(X_test)
print(y_pred)
#check Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
#test with New flower data
new_flower=[[5.1,3.5,1.4,0.2]]
prediction=model.predict(new_flower)
print("Predicted class: ",iris.target_names[prediction])
#save the model 
import joblib
joblib.dump(model,"iris_model.pkl")
print("Model saved successfully")