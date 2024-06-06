# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = 0
random_model=RandomForestClassifier(n_estimators=100,n_jobs=-1)
random_model.fit(X_train,y_train)
log_model=LogisticRegression(n_jobs=-1)
log_model.fit(X_train,y_train)

def perdiction(model,sw,sl,pw,pl):
    global score
    answer=model.predict([[sl,sw,pl,pw]])
    score = model.score(X_train, y_train)
    if answer[0]==0:
            return('Iris-setosa')
    elif answer[0]==1:
            return('Iris-virginica')
    elif answer[0]==2:
            return('Iris-versicolor')

st.sidebar.title('Iris Flower Prediction App')
sepall=st.sidebar.slider('Sepal Length',0.0,10.0)
sepalw=st.sidebar.slider('Sepal Width',0.0,10.0)
petall=st.sidebar.slider('Petal Length',0.0,10.0)
petalw=st.sidebar.slider('Petal Width',0.0,10.0)

dropdown=st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
predict=st.sidebar.button('Predict')

if predict == True:
    answer=0
    if dropdown == 'Support Vector Machine':
        answer=perdiction(svc_model,sepall, sepalw, petall, petalw)

    elif dropdown == 'Logistic Regression':
        answer = perdiction(random_model, sepall, sepalw, petall, petalw)
    elif dropdown == 'Random Forest Classifier':
        answer = perdiction(log_model, sepall, sepalw, petall, petalw)

    st.write('flower is ', answer)
    st.write('the score is ', score)



