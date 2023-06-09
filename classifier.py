import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# calculate the accuracy of the predictions
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# import all the models used here
"""
    if model == 'Random Forest Classifier':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model == 'Logistic Regression':
        clf = LogisticRegression(random_state=42)
    elif model == 'Support Vector Classifier':
        clf = SVC(random_state=42)
    elif model == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model == 'Decision Tree Classifier':
        clf = DecisionTreeClassifier(random_state=42)
    elif model == 'Gaussian Naive Bayes':
        clf = GaussianNB()
    elif model == 'Linear Discriminant Analysis':
        clf = LinearDiscriminantAnalysis()
    elif model == 'Quadratic Discriminant Analysis':
        clf = QuadraticDiscriminantAnalysis()
    elif model == 'AdaBoost Classifier':
        clf = AdaBoostClassifier(random_state=42)
    elif model == 'Gradient Boosting Classifier':
        clf = GradientBoostingClassifier(random_state=42)
    elif model == 'Extra Trees Classifier':
        clf = ExtraTreesClassifier(random_state=42)
    elif model == 'Bagging Classifier':
        clf = BaggingClassifier(random_state=42)
    elif model == 'MLP Classifier':
        clf = MLPClassifier(random_state=42)
    elif model == 'XGBoost Classifier':
        clf = XGBClassifier(random_state=42)
    elif model == 'CatBoost Classifier':
        clf = CatBoostClassifier(random_state=42)
    elif model == 'LightGBM Classifier':
        clf = LGBMClassifier(random_state=42)
    elif model == 'NuSVC':
        clf = NuSVC(random_state=42)

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier



# Define a function to extract x,y coordinate pairs from path commands
def get_coordinates(path_commands):
    current_point = [0,0]
    coordinates = []
    for command in path_commands:
        if command[0] == 'M':
            current_point = command[1:]
            coordinates.append(current_point)
        elif command[0] == 'L':
            current_point = command[1:]
            coordinates.append(current_point)
        elif command[0] == 'Q':
            control_point = command[1:3]
            end_point = command[3:]
            for t in range(1, 11):
                t_normalized = t / 10
                x = (1 - t_normalized) ** 2 * current_point[0] + 2 * t_normalized * (1 - t_normalized) * control_point[0] + t_normalized ** 2 * end_point[0]
                y = (1 - t_normalized) ** 2 * current_point[1] + 2 * t_normalized * (1 - t_normalized) * control_point[1] + t_normalized ** 2 * end_point[1]
                coordinates.append([x, y])
            current_point = end_point
    return coordinates


def classify(img, path='labels.csv'):
    path_commands = pd.read_csv(path)
    coordinates = pd.DataFrame()
    #path_commands["path"][1]
    for i in range(len(path_commands)):
        # path is located in path_commands['path'][i]
        id, class_, path = i,path_commands["stroke"][i], get_coordinates(ast.literal_eval(path_commands["path"][i]))
        # append the coordinates to the dictionary as class_, path key, value pair
        coordinates[i] = [class_, path]

    # get x, y value pairs from coordinates
    x_coords = []
    y_coords = []
    for key, value in coordinates.items():
        for x, y in value[1]:
            x_coords.append(x)
            y_coords.append(y)

    # transpose coordinates
    coordinates = coordinates.T
    #coordinates
    coordinates.columns = ["class", "path"]
    # separate the values in the path column into individual rows 
    coordinates = coordinates.explode("path")

    # search the x,y coordinates given in the path column in the image and get the pixel values assigned to a new column "pixel_value"
    coordinates["pixel_value"] = coordinates["path"].apply(lambda x: img.getpixel((x[0], x[1])))


    classes = coordinates["class"].unique()
    classes = list(classes)

    # convert the classes in the class column to integers from the list classes to 0,1,...,n
    coordinates["class"] = coordinates["class"].apply(lambda x: classes.index(x))
    # split the path column into x and y coordinates
    coordinates["x"] = coordinates["path"].apply(lambda x: x[0])
    coordinates["y"] = coordinates["path"].apply(lambda x: x[1])
    dataset = coordinates[["pixel_value", "class"]]
    return dataset 



def train_model(dataset, model):
    X = dataset["pixel_value"]
    y = dataset["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model == 'Random Forest Classifier':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model == 'Logistic Regression':
        clf = LogisticRegression(random_state=42)
    elif model == 'Support Vector Classifier':
        clf = SVC(random_state=42)
    elif model == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model == 'Decision Tree Classifier':
        clf = DecisionTreeClassifier(random_state=42)
    elif model == 'Gaussian Naive Bayes':
        clf = GaussianNB()
    elif model == 'Linear Discriminant Analysis':
        clf = LinearDiscriminantAnalysis()
    elif model == 'Quadratic Discriminant Analysis':
        clf = QuadraticDiscriminantAnalysis()
    elif model == 'AdaBoost Classifier':
        clf = AdaBoostClassifier(random_state=42)
    elif model == 'Gradient Boosting Classifier':
        clf = GradientBoostingClassifier(random_state=42)
    elif model == 'Extra Trees Classifier':
        clf = ExtraTreesClassifier(random_state=42)
    elif model == 'Bagging Classifier':
        clf = BaggingClassifier(random_state=42)
    elif model == 'MLP Classifier':
        clf = MLPClassifier(random_state=42)
    elif model == 'XGBoost Classifier':
        clf = XGBClassifier(random_state=42)
    elif model == 'CatBoost Classifier':
        clf = CatBoostClassifier(random_state=42)
    

    # elif model == 'LightGBM Classifier':
    #     clf = LGBMClassifier(random_state=42)
    # elif model == 'NuSVC':
    #     clf = NuSVC(random_state=42)



    clf.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred = clf.predict(X_test.values.reshape(-1, 1))
    test_acc = accuracy_score(y_test, y_pred)

    return clf, test_acc

def train_model_RF(dataset):



    X = dataset["pixel_value"]
    y = dataset["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # reshape the data to fit the model
    rf.fit(X_train.values.reshape(-1, 1), y_train)
    #rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test.values.reshape(-1, 1))

    accuracy = accuracy_score(y_test, y_pred)


    return rf, accuracy

def predict(model, img):
    # load the image
    #img = Image.open(img)

    #plt.imshow(img, cmap="gray")
    # use the model to predict the class of the pixel values in the image
    # get the pixel values from the image
    pixel_values = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixel_values.append(img.getpixel((x, y)))

    # reshape the pixel values to fit the model
    pixel_values = np.array(pixel_values).reshape(-1, 1)

    # predict the class of the pixel values
    print("Predicting...")
    
    predictions = model.predict(pixel_values)
    predictions_image = predictions.reshape(img.size[0], img.size[1])
    #mirror the predictions image
    #
    #rotate 90 degrees to the right
    predictions_image = np.rot90(predictions_image, k=1)    
    predictions_image = np.flip(predictions_image, axis=0)

    return predictions_image

