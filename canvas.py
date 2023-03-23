import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import classifier as cf 
import numpy as np 
from skimage import color, io

# # Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing Tool", ("freedraw",)
)

# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("freedraw")
# )

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
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
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

    model_choosen = st.selectbox('Choose a model', ('K-Nearest Neighbors', 'Random Forest Classifier',  'Support Vector Classifier', 'Logistic Regression', 'DecisionTreeClassifier', 'GaussianNB', 'MLPClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier'))

    if st.button('Predict'):


        dataset = cf.classify(img) 

        model, score = cf.train_model_RF(dataset)
        # model, scores = cf.train_model(dataset)
        predictions_image = cf.predict(model, img)

        # display the predictions
        st.write("Predicting...")
        # write unique values iin the predictions
        #st.write("Unique values in the predictions: ", predictions_image)
        # display the predictions 
        # write a function to convert the predictions_image from integer to a random color

        # def color_map(predictions_image):
        #     colors = {0: (0,0,0), 1: (255,255,0), 2: (0,255,0), 3: (255,0,0)}
        #     # create a new array of the same size as the predictions_image
        #     predictions_image = np.zeros((predictions_image.shape[0], predictions_image.shape[1], 3))
        #     # loop through the predictions_image and replace the values with the colors
        #     for i in range(predictions_image.shape[0]):
        #         for j in range(predictions_image.shape[1]):
        #             predictions_image[i,j] = colors[predictions_image[i,j]]
        #     return predictions_image
        
        #predictions_image = color_map(predictions_image)
        rgb_image = color.label2rgb(predictions_image)
        st.image(img)
        st.write("Model: ", model_choosen, "Score: ", score)
        st.image(rgb_image)




