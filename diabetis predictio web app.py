import numpy as np
import pickle
import streamlit as st


# loading the saved model
scaler, loaded_model = pickle.load(open('C:/diabeties/full diabetis predictive project/trained_model.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):

  # change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    Prediction = loaded_model.predict(input_data_reshaped)
    print (Prediction)
    
    if Prediction[0] == 1:
        return "⚠️ The model predicts that you may have diabetes"
    else:
        return "✅ The model predicts you are not likely to have diabetes"


def main ():
    
    st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.markdown("""
# 🩺 Diabetes Prediction Web App
Welcome! This tool uses machine learning to predict the likelihood of diabetes based on input health data.
""")
st.markdown("---")

# getting input data

with st.form("prediction_form"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        Pregnancies = st.text_input("Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure")
        SkinThickness = st.text_input("Skin Thickness")
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI")
        DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function")
        Age= st.text_input("Age")

  
    # code for prediction
    diagnosis = ''

    # creating a button for prediction 

    if st.form_submit_button('Diabetis test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
with st.sidebar:
    st.markdown("### 👨‍💻 About the Developer")
    st.markdown("Built by Ajala Boluwatife as a final-year project.")
    st.markdown("[View GitHub Repo](https://github.com/AjalaBolu/Diabetis_machine_learning_project.git)")

    
if __name__ == '__main__' :
    main()