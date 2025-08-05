import numpy as np
import pickle
import streamlit as st

# Load the saved model
scaler, loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
   
    # change the input data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  # standardizing the input data
  std_data =scaler.transform(input_data_reshaped)
  print(std_data)

  Prediction = loaded_model.predict(std_data)
  print (Prediction)


  if ( Prediction[0] == 0 ):
    return 'The person is not likely to be diabetic'
  else:
    return 'The person is likely going to be diabetic'

# Main function
def main():

    st.title("""
     🩺 Diabetes Prediction Web App
    Welcome! This tool uses machine learning to predict the likelihood of diabetes based on input health data.
    """)
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI level")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    # code for prediction
    diagnosis = ''
    
    if st.button('Diabetis test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
        st.success(diagnosis)
        


# Run the app
if __name__ == '__main__':
    main()


