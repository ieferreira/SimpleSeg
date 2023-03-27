import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import classifier as cf 
import numpy as np 
from skimage import color, io

#  Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing Tool", ("freedraw",)
)


stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color  = st.sidebar.selectbox(
    "Segmentation Class", ("#000","#FF0", "#0F0","#F00")
)
bg_color = "#eee"
bg_image = st.sidebar.file_uploader("Training image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

    
img = Image.open(bg_image) if bg_image else None



if bg_image is not None:
    height, width = img.size


    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image= img,
        update_streamlit=realtime_update,
        height=width,
        width = height, 
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Do something interesting with the image data and paths

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        #st.dataframe(objects)
        #st.dataframe(objects["path"])
    # save path as a csv
    if st.button('Save Labels'):
        objects.to_csv('labels.csv')

        # add a dropdown to choose the model

    model_choosen = st.selectbox('Choose a model', ('K-Nearest Neighbors',"XGBoost Classifier", 'Random Forest Classifier',  'Support Vector Classifier', 'Logistic Regression', 'Decision Tree Classifier', 'Gaussian Naive Bayes', 'MLP Classifier', 'AdaBoost Classifier', 'GradientBoostingClassifier', 'XGBoost Classifier', 'CatBoost Classifier'))

    if st.button('Predict'):


        dataset = cf.classify(img) 

        model, score = cf.train_model(dataset, model_choosen)
        #model, score = cf.train_model_RF(dataset)

        # model, scores = cf.train_model(dataset)
        predictions_image = cf.predict(model, img)

        # display the predictions
        st.write("Predicting...")

        st.write("Predictions: ", predictions_image)
        rgb_image = color.label2rgb(predictions_image)
        st.image(img)
        st.write("Model: ", model_choosen, "Score: ", score)
        st.image(rgb_image)




