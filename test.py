# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 07:35:32 2022

@author: venki
"""
import pickle
import streamlit as st
import pandas as pd
# loading the trained model
#df = pd.read_csv("/home/venky/Desktop/Datascience_360/Real_Project_costprediction/VS-Edtech_CourseCost_Model-Deployment/data.csv")
pickle_in = open("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\VS-Deploy-11-07\\data_model.pkl", 'rb') 
model = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Institute,Subject,Location, Trainer_Qualification, Trainer_experiance, Online_classes, Offline_classes, Course_hours,
                  Course_level,Course_rating,Rental_permises, Trainer_slary,Maintaince_cost,Non_teaching_staff_salary, Placements):
    # Pre-processing user input       
    if Institute == "360DigiTMG":
       Institute = 0
    elif Institute == "innomatics":
       Institute = 13
    elif Institute == "Great_Learn":
       Institute = 8   
    elif Institute == "Coursera":
       Institute = 1   
    elif Institute == "Datatrain":
       Institute = 3
    elif Institute == "Upgrad":
       Institute = 12  
    elif Institute == "ExcelR":
       Institute = 7
    elif Institute == "Simple_learn":
       Institute = 10   
    elif Institute == "Udemy":
       Institute = 11 
    elif Institute == "Edvancer":
       Institute = 5
    elif Institute == "Edureka":
       Institute = 4
    elif Institute == "Guvi":
       Institute = 9 
    elif Institute == "Data_camp":
       Institute = 2
    elif Institute == "Edx":
       Institute = 6
    

    if Subject == "python":
        Subject = 9
    elif Subject == "ProjectManagement":
        Subject = 8
    elif Subject == "DataScience":
        Subject = 3
    elif Subject == "ArtificialIntelligence":
        Subject = 0
    elif Subject == "FullstackDataScience":
        Subject = 7
    elif Subject == "BigData":
        Subject = 1
    elif Subject == "CloudComputing":
        Subject = 2
    elif Subject == "DigitalTransformation":
        Subject = 6
    elif Subject == "DigitalMarketing":
        Subject = 5
    elif Subject == "Dataanalysis":
        Subject = 4
    
    if Location == "Bengaluru":
        Location = 1
    elif Location == "Hyderabad":
        Location = 5
    elif Location == "Chennai":
        Location = 3
    elif Location == "Ahmedabad":
        Location = 0
    elif Location == "Pune":
        Location = 8
    elif Location == "Delhi":
        Location = 4
    elif Location == "Calcutta":
        Location = 2
    elif Location == "Kanpour":
        Location = 6
    elif Location == "Mumbai":
        Location = 7
    
 
    if Trainer_Qualification == "Graduate":
        Trainer_Qualification = 1
    elif Trainer_Qualification == "UnderGraduate":
        Trainer_Qualification = 2
    elif Trainer_Qualification == "Doctorate":
        Trainer_Qualification = 0
    
    if Online_classes == "Yes":
        Online_classes = 0
    else:
        Online_classes = 1
    
    if Offline_classes == "Yes":
        Offline_classes = 0
    else:
        Offline_classes = 1
        
    if Course_level == "beginer":
        Course_level = 1
    elif Course_level == "intermediate":
        Course_level = 2
    elif Course_level == "advanced":
        Course_level = 0
        
    if Placements == "Yes":
         Placements = 0
    else:
        Placements = 1
        
        
    # Making predictions 
    prediction = model.predict(pd.DataFrame([[Institute,Subject,Location, Trainer_Qualification, Trainer_experiance, Online_classes, Offline_classes, Course_hours,
          Course_level,Course_rating,Rental_permises, Trainer_slary,Maintaince_cost,Non_teaching_staff_salary, Placements]]))
    return prediction
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Price Prediction of an EdTech Product </h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    #Institute_Name
    #company = st.selectbox('Brand', df['Company'].unique())
    Institute = st.selectbox('Institute_name:',['innomatics','Great_Learn','Coursera','Datatrain','Upgrad','ExcelR','Simple_learn','Udemy','Edvancer','Edureka','Guvi','Data_camp','Edx','360DigiTMG'])

    #Subject_Name
    Subject = st.selectbox("Subject_name:",['ProjectManagement', 'python','DataScience','ArtficialIntelligence','FullstackDataScience','BigData','CloudComputing','DigitalTransformation','DigitalMarketing','Dataanalysis'])

    #Location
    Location = st.selectbox("Location:",['Bengaluru','Hyderabad','Chennai','Ahmedabad','Pune','Delhi', 'calcutta','Kanpour','Mumbai'])

    ## Trainer
    Trainer_Qualification = st.selectbox("Trainer_Qualification:",['UnderGraduate','Graduate','Doctorate'])
    Trainer_experiance = st.sidebar.slider('Trainer_experiance:', 1, 20, 1 )
    ## Classes
    Online_classes = st.selectbox("Online_classes:",['Yes','No'])
    Offline_classes = st.selectbox("Offline_classes:",['Yes','No'])

    # Hours
    Course_hours = st.sidebar.slider('Course_hours', 40, 200)
    Course_level = st.selectbox("Course_level:",['intermediate','beginer','advanced'])
    Course_rating = st.sidebar.slider('Course_rating:', 1, 5)
    ## Maintaince cost
    Rental_permises = st.sidebar.slider('Rental_permises:', 15, 150)
    Trainer_slary = st.sidebar.slider('Trainer_slary:', 200, 800)
    Maintaince_cost = st.sidebar.slider('Maintaince_cost', 15, 150)
    Non_teaching_staff_salary = st.sidebar.slider('Non_teaching_staff_salary:', 50, 225)

    # placements
    Placements =  st.selectbox('Placements:',['Yes','No'])
    
    result =""
    
    # Converting Features into DataFrame

    #features_df  = pd.DataFrame([prediction])
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button('Price'):
       result = prediction(Institute,Subject,Location, Trainer_Qualification, Trainer_experiance, Online_classes, Offline_classes, Course_hours,
                          Course_level,Course_rating,Rental_permises, Trainer_slary,Maintaince_cost,Non_teaching_staff_salary, Placements)
       
       st.success(f'The predicted price of the course is {result[0]:.0f} INR(per hour)')
    
     
if __name__=='__main__': 
    main()