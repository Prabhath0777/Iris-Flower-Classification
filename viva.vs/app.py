import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)


st.title(" Mini-Iris Flower Classification")
st.write("Enter flower measurements and get predicted species.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

sample = [[sepal_length, sepal_width, petal_length, petal_width]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
species = iris.target_names[prediction]

st.subheader("Prediction")
st.write(f"The predicted species is: **{species.capitalize()}**")
st.write(f"Accuracy is: **{accuracy}**")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

pd.set_option("display.max_rows", None)  
st.dataframe(df)