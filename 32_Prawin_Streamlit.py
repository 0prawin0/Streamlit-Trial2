import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score

st.sidebar.header("This is a web app to check accuracy and F1 score of each algorithm on Iris dataset")

iris = sns.load_dataset('iris')

df = iris.copy()

df['y'] = df['species'].map({'setosa': 1, 'versicolor': 0, 'virginica': 0})
df.drop('species',inplace=True,axis=1)

y = df['y']
X = df.drop('y',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

lr = LogisticRegression()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

model_lr=lr.fit(X_train, y_train)
model_dt=dt.fit(X_train, y_train)
model_knn=knn.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)
pred_dt = model_lr.predict(X_test)
pred_knn = model_lr.predict(X_test)

acc_lr = accuracy_score(y_test,pred_lr)
acc_dt = accuracy_score(y_test,pred_dt)
acc_knn = accuracy_score(y_test,pred_knn)

f1_lr = f1_score(y_test,pred_lr)
f1_dt = f1_score(y_test,pred_dt)
f1_knn = f1_score(y_test,pred_knn)

st.write("Accuracy of Logistic Regression is ", acc_lr)
st.write("F1 score of Logistic Regression is ", f1_lr)
st.write("Accuracy of Decistion Tree is ", acc_dt)
st.write("F1 score of Decistion Tree is ", f1_dt)
st.write("Accuracy of KNN Regressor is ", acc_knn)
st.write("F1 score of KNN Regressor is ", f1_knn)

