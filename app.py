import streamlit as st

import joblib

model_nb = joblib.load('Fake News Detection')
vect = joblib.load('vect.pkl')

def main():
  st.title('Fake News Prediction') #creates a title in web app
  ip = st.text_input('Enter text:') #creates a text box in web app
  if st.button('Predict'):
    data=[ip]
    cv=vect.transform(data).toarray()
    prediction=model_nb.predict(cv)
    result=prediction[0]
    if result==0:
      st.error("Fake News")
    else:
      st.success("Not a Fake News")
   
main()  
