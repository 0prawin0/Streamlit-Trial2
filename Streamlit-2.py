import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,f1_score

st.sidebar.header("This is a web app to check accuracy and F1 score of each algorithm on Iris dataset")

iris = sns.load_dataset('iris')

df = iris.copy()

df['y'] = df['species'].map({'setosa': 1, 'versicolor': 0, 'virginica': 0})
df.drop('species',inplace=True,axis=1)

y = df['y']
X = df.drop('y',axis=1)

lr = LogisticRegression()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

model_lr=lr.fit(X, y)
model_dt=dt.fit(X, y)
model_knn=knn.fit(X, y)

pred_lr = model_lr.predict(X)
pred_dt = model_lr.predict(X)
pred_knn = model_lr.predict(X)

acc_lr = accuracy_score(y,pred_lr)
acc_dt = accuracy_score(y,pred_dt)
acc_knn = accuracy_score(y,pred_knn)

f1_lr = f1_score(y,pred_lr)
f1_dt = f1_score(y,pred_dt)
f1_knn = f1_score(y,pred_knn)

# st.sidebar.header("Choose the algorithm")
# selected_option = st.sidebar.selectbox("Select an Option", ["Logistic Regression", "Decision Tree ", "KNN "])

# if selected_option == 'Logistic Regression':
#     st.write("Accuracy of Logistic Regression is ", acc_lr)
#     st.write("F1 score of Logistic Regression is ", f1_lr)
# elif selected_option == 'Decision Tree ':
#     st.write("Accuracy of Decistion Tree is ", acc_dt)
#     st.write("F1 score of Decistion Tree is ", f1_dt)
# else:
#     st.write("Accuracy of KNN Regressor is ", acc_knn)
#     st.write("F1 score of KNN Regressor is ", f1_knn)

st.write("Accuracy of Logistic Regression is ", acc_lr)
st.write("F1 score of Logistic Regression is ", f1_lr)
st.write("Accuracy of Decistion Tree is ", acc_dt)
st.write("F1 score of Decistion Tree is ", f1_dt)
st.write("Accuracy of KNN Regressor is ", acc_knn)
st.write("F1 score of KNN Regressor is ", f1_knn)

