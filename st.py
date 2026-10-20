import streamlit as st
import dill
import numpy as np
from sklearn.preprocessing import StandardScaler
with open('model.pkl','rb') as f:
    model=dill.load(f)

st.header('Diabetes Test')
Pregnancies=st.number_input('Pregnancy',step=1,min_value=0,max_value=17)
Glucose=st.number_input('Glucose',max_value=199,min_value=0)
BloodPressure=st.number_input('BloodPressure',min_value=0,max_value=122)
SkinThickness=st.number_input('SkinThickness',min_value=0,max_value=99)
Insulin=st.number_input('Insulin',min_value=0,max_value=846)
BMI=st.number_input('BMI',max_value=67.1,min_value=0.0)
DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction',min_value=0.07800,max_value=2.42000)
Age=st.number_input('Age',min_value=21,max_value=81,step=1)
ss=StandardScaler()
def prediction(p,g,bp,s,i,bmi,dp,a):
    input_data=(p,g,bp,s,i,bmi,dp,a)
    input_data_reshaped=np.array(input_data).reshape(1,-1)
    std_data=ss.fit_transform(input_data_reshaped)
    prediction=model.predict(std_data)
    print(prediction)
    if prediction[0]==0:
        st.success('Person is not diabetic')
    else:
        st.warning('Person is diabetic')

if st.button('Diabetes Test'):
    prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)