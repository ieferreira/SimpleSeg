Project Name: Hyperspectral and Grayscale Image Streamlit App
This project aims to develop a Streamlit app that can classify hyperspectral and grayscale images using multiple models. The app will allow the user to draw on an image canvas and save the labels as a CSV file. The user can also load an image file and predict the class of each pixel in the image.

Folder Structure
00_proto.ipynb: A Jupyter notebook containing the initial prototype of the app.
00_proto_2.ipynb: A Jupyter notebook containing the updated version of the prototype.
classifier.py: A Python file containing the functions to extract x, y coordinate pairs from path commands, classify an image, train a model, and predict the class of pixel values in an image.
petroseg.py: A Python file containing the implementation of the simple 1D convolutional neural network for classifying a single label from an array of 241 values.
canvas.py: A Python file containing the implementation of the Streamlit canvas.
img.png: A sample image for testing the app.
path.csv: A sample CSV file containing path commands and corresponding class labels.
labels.csv: A CSV file where the user can save the labels drawn on the canvas.
How to Use
Clone the repository to your local machine.
Install the required packages using pip install -r requirements.txt.
Run streamlit run 00_proto_2.ipynb to start the app.
Select the drawing tool and adjust the stroke width and color.
Draw on the canvas and save the labels by clicking the "Save Labels" button.
Load an image file by clicking the "Training image" button and selecting an image.
Click the "Predict" button to predict the class of each pixel in the image.
The predicted image will be displayed below the original image.
Credits
OpenCV, scikit-image, and NumPy for image processing and manipulation.
Streamlit for building the app.
Random Forest Classifier for classification.
Simple 1D Convolutional Neural Network for classifying a single label from an array of 241 values.