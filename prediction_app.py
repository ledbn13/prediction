import numpy as np
import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os

option = st.sidebar.selectbox('What do you want?', (' ','Get the predicted extinction spectrum', 'Upload your extinction spectrum'),index=0)

if option == ' ':
    st.sidebar.write('Choose an option')

elif option == 'Upload your extinction spectrum':
   os.makedirs("tempDir", exist_ok=True)
   image_file = st.file_uploader("Please upload your extinction spectrum here.", type=["png","jpeg","jpg"])
   if image_file is not None:
       image = Image.open(image_file)
       st.image(image)
       file_details = {"FileName": image_file.name, "FileType": image_file.type}
       st.write(file_details)
       with open(os.path.join("tempDir", image_file.name), "wb") as f: 
         f.write(image_file.getbuffer())         
       st.success("Saved File")

else:
# loading the saved model
    loaded_model = pickle.load(open('trained_model.sav','rb'))

# creating a function for Prediction
    def extinction_prediction(input_data):
        prediction = loaded_model.predict(input_data)
        xplot = np.linspace(300,1201,901)
        fig, ax = plt.subplots()
        return prediction, xplot, fig

    st.title('Extinction spectrum prediction')
    st.write('This is web app to predict the extinction spectrum of nanostructured materials based on several features. Please choose the value of each feature and click on the Predict Extinction spectrum button to get the result.')
    st.write('---')

# getting the input data from the user
# Radius
    R = st.number_input('Radius (nm)', min_value=10)

# code for Prediction
    extinction = []
    xplot = []
    
# creating a button for Prediction
    if st.button('Predict Extinction spectrum'):
        input = [[R, i] for i in range(300,1201)]
        extinction, xplot,fig = extinction_prediction(input)
        xplot = np.linspace(300,1201,901)
        fig, ax = plt.subplots()
        ax.plot(xplot,extinction)
        fig.suptitle('Extinction spectrum of {} nm gold nanoparticle'.format(R))
        ax.tick_params(direction='in')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel('extinction cross section (nm^2)')
        ax.set_xlim(300,1200)
        st.pyplot(fig)

        if st.button('Download the plot'):
            fig.savefig('extinction_spectrum.png')
            with open('extinction_spectrum.png', 'rb') as file:
                st.download_button('Download the plot', data=file, file_name='extinction_spectrum.png', mime='image/png')
