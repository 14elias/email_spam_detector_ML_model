
from fastapi import FastAPI
import joblib
import streamlit as st

app = FastAPI()
model = joblib.load("spam_classifier.pkl")

# @app.post("/")
# def index(text: str):
#     prediction = ""
#     input_text = ""
#     result = model.predict([text])[0]
#     prediction = "Spam" if result == 1 else "Not Spam"

#     return prediction

def runer():
    prediction = ""
    input_text = ""
    st.title('Emal Detector')
    input_text = st.text_area(label='input the email to be detected')

    button = st.button('Evaluate')
    if input_text:
        if button:
            result = model.predict([input_text])[0]
            prediction = "Spam" if result == 1 else "Not Spam"
            st.write(prediction)
    return prediction


if __name__ == '__main__':
    runer()
