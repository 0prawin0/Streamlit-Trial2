import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error

X1 = [[781],[702],[775],[542],[528],[466],[800],[745],[460],[240],[592],[539],[705],[515],[727],[887],[976],[654],[235],[775]]
X2 = [[1],[0],[0],[0],[0],[1],[0],[0],[1],[0],[1],[0],[1],[0],[0],[1],[1],[1],[0],[1]]
y = [[61],[54],[96],[76],[58],[65],[33],[76],[52],[42],[44],[61],[75],[51],[10],[82],[13],[18],[89],[58]]
data = {'X1':X1,'X2':X2,'y':y}

df = pd.DataFrame(columns = ['X1','X2','y'],data = data)
y = df['y']
X = df.drop('y')

model_lr = LinearRegression()
model_dt = DecisionTreeRegressor()
model_knn = KNeighborsRegressor()

model_lr.fit(X, y)
model_dt.fit(X, y)
model_knn.fit(X, y)

pred_lr = model_lr.predict(X)
pred_dt = model_lr.predict(X)
pred_knn = model_lr.predict(X)

mse_lr = mean_squared_error(y,pred_lr)
mse_dt = mean_squared_error(y,pred_dt)
mse_knn = mean_squared_error(y,pred_knn)

mape_lr = mean_absolute_percentage_error(y,pred_lr)
mape_dt = mean_absolute_percentage_error(y,pred_dt)
mape_knn = mean_absolute_percentage_error(y,pred_knn)

st.sidebar.header("Choose the algorithm")
selected_option = st.sidebar.selectbox("Select an Option", ["Linear Regression", "Decision Tree Regressor", "KNN Regressor"])

if selected_option == 'Linear Regression':
    st.write("MAPE of Linear Regression is ", mape_lr)
    st.write("MSE of Linear Regression is ", mse_lr)
elif selected_option == 'Decision Tree Regressor':
    st.write("MAPE of Decistion Tree is ", mape_dt)
    st.write("MSE of Decistion Tree is ", mse_dt)
else:
    st.write("MAPE of KNN Regressor is ", mape_knn)
    st.write("MSE of KNN Regressor is ", mse_knn)

