Dysgraphia Web App via MSF-EWS Fusion

Project Overview
This project processes raw handwriting sensor data using A Streamlit-based web application that analyzes and detects handwriting patterns associated with Dysgraphia using handcrafted features (surface and in-air features) and deep image feature.

Key Features
1. Handcrafted Feature Extraction 
2. Deep Feature Extraction

Data Output & Integration
- Processes raw sensor CSV/TXT files (minimum 7 data columns required).
- Generates a structured 2D feature array (N, feature)` ready for model input.
- Allows downloading extracted features in `.npy` format for direct compatibility with Scikit-Learn pipelines.

Input Data Format
1. The input CSV/TXT file must contain at least 7 ordered columns:
    - `X` - Horizontal Position
    - `Y` - Vertical Position
    - `T` - Timestamp
    - `Status` - Pen Down (1) / Pen Up (0)
    - `Pressure` - Pen Pressure
    - `Altitude` - Altitude Angle
    - `Azimuth` - Azimuth Angle
2. The input image 

How to Run Locally
Step 1: Open Command Prompt (CMD) in the Folder
1.	Press the Windows key + R on your keyboard, type `cmd`, and press Enter.
2.	Switch to the D: drive by typing this command and pressing Enter: D:
4.	Navigate to your project folder: cd D:\web-disgrafia
Step 2: Install Libraries & Run the Web App
1.	Install the required libraries (run this only once): pip install -r requirements.txt
2.	Run the Streamlit application: streamlit run app.py

Note
1.	Make sure Python is installed.
2.	If the torch module is not installed, type in cmd: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 
3.	Install the necessary libraries such as: pip install pillow scikit-learn scipy matplotlib seaborn pandas or pip install -r additionalrequirements.txt

