import streamlit as st
import acquisition
import sensorfeature
import imagefeature
import classification

st.title("Dysgraphia Application Dashboard : EWS Fusion")

# Membuka 4 tab navigasi
tab1, tab2, tab3, tab4 = st.tabs([
    "Acquisition", 
    "Sensor Feature Extraction", 
    "Image Feature Extraction", 
    "Fusion Classification"
])

with tab1:
    acquisition.run()

with tab2:
    sensorfeature.run()

with tab3:
    imagefeature.run()

with tab4:
    classification.run() 