import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# load the iris dataset
iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#parameter for  the randomforest 
max_depth = 4
n_estimators = 20

# apply mlflow
mlflow.set_experiment("iris-dt")
with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("confusion_matrix")

    #save the plot as artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(dt,'decision tree')

    mlflow.set_tag('author','Abhishek')
    mlflow.set_tag('model','Decision Tree')

    print('accuracy : ',accuracy)
