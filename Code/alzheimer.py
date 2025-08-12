
# Importing libraries
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from skimage.feature import hog
from skimage.transform import resize

# Reading an image
read = cv2.imread(f'../Dataset/Axial/CN/CN002_S_0413a084.png',cv2.IMREAD_UNCHANGED)

# Converting the color of image from BGR to RGB
cvt = cv2.cvtColor(read,cv2.COLOR_BAYER_BG2RGB)

# Displaying image
plt.imshow(cvt)

# Resizing the image size into (150,150)
resize_img = resize(cvt,(150,150))

# Image after resizing
plt.imshow(resize_img)

# Assigning height, width, train directory and class names
IMAGE_HEIGHT , IMAGE_WIDTH = 150, 150
CLASSES_LIST = ['AD','CI','CN']
train_dir = "../Dataset/Axial"

# Checkin length of each class
n_train_demented = len(os.listdir(f'{train_dir}/AD'))
n_train_mild = len(os.listdir(f'{train_dir}/CI'))
n_train_non = len(os.listdir(f'{train_dir}/CN'))

# Total length of the whole classes combined
total_train = n_train_mild + n_train_demented + n_train_non

# Count length
count = [n_train_demented, n_train_mild, n_train_non, total_train]
length = ['demented', ' mild demented', 'non demented', 'total images']

for i, sum in zip(length, count):
  print(f'Total counts of {i} is {sum}')
  print(100*'-')

# Define the class labels and their respective counts
class_labels = ['Demented', 'Mild Cognitive Impairment', 'Non-Demented']
class_counts = [n_train_demented, n_train_mild, n_train_non]

# Create a countplot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=class_labels, y=class_counts, palette="viridis")

# Annotate the counts on top of the bars
for i, count in enumerate(class_counts):
    ax.text(i, count + 2, str(count), ha='center', va='bottom', fontsize=12)

plt.title("Distribution of Classes in Training Data")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Resizing and combining data
roi_top_left = (50,50)
roi_bottom_right = (100,100)

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

hog_features = []
roi_features = []
label = []
for class_name in os.listdir(train_dir):
    class_folder = os.path.join(train_dir, class_name)

    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)
        image = cv2.imread(image_path)
        resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # ROI extraction
        roi =  resized[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]


        # Compute HOG features for the ROI
        hog_feature = hog(roi, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False,channel_axis=2)


        hog_features.append(hog_feature)
        roi_features.append(roi)
        label.append(class_name)

# Convert lists to numpy arrays
X_roi = np.array(roi_features)
X_hog = np.array(hog_features)
y = np.array(label)

# ROI feature extracted
print('ROI features\n',X_roi)

# Reshaping to 2D
X_roi_reshaped = X_roi.reshape(X_roi.shape[0], -1)

# HOG feature extracted

print('HOG features\n', X_hog)

# Combining HOG and ROI features
multi_view_representation = np.hstack(( X_hog,X_roi_reshaped))
print('Multi View Representation\n',multi_view_representation)

# Features
X = multi_view_representation

# Labels
le = LabelEncoder()
y = le.fit_transform(y)
print('Class Label Encoded',y)

#Train Test Splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# SVM
svm_clf = SVC()
svm_clf.fit(X_train,y_train)

print(f'SVM Training Score: {svm_clf.score(X_train,y_train)}')
print(f'SVM Testing Score: {svm_clf.score(X_test,y_test)}')

svm_y_pred = svm_clf.predict(X_test)

svm_acc = accuracy_score(y_test,svm_y_pred)

print(f'SVM Accuracy Score: {svm_acc}')

# Confusion Matrix
svm_cm = confusion_matrix(y_test,svm_y_pred)
print(f'SVM Confusion Matrix: {svm_cm}')

# Plot SVM Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(svm_cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Preidicted Classes')
plt.ylabel('Actual Classes')
plt.title('SVM Confusion Matrix')
plt.show()

# DecisionTree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)

# Evaluation
print(f'DecisionTree Training Score: {dt_clf.score(X_train,y_train)}')
print(f'DecisionTree Testing Score: {dt_clf.score(X_test,y_test)}')

dt_y_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test,dt_y_pred)
print(f'DecisionTree Accuracy Score: {dt_acc}')

# Confusion Matrix
dt_cm = confusion_matrix(y_test,dt_y_pred)
print(f'DecisionTree Confusion Matrix: {dt_cm}')

# Plot Decision Tree Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(dt_cm,annot=True,fmt='d',cmap='BuGn')
plt.xlabel('Preidicted Classes')
plt.ylabel('Actual Classes')
plt.title('DecisionTree Confusion Matrix')
plt.show()

# SVM-2K

# Number of classes
num_classes = len(set(y_train))

# Initialize a list to store binary SVM classifiers
binary_classifiers = []

# Train binary SVM classifiers for each pair of classes (i, j)
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        # Create a binary dataset containing samples from classes i and j
        binary_X_train = []
        binary_y_train = []

        for k in range(len(X_train)):
            if y_train[k] == i:
                binary_X_train.append(X_train[k])
                binary_y_train.append(1)
            elif y_train[k] == j:
                binary_X_train.append(X_train[k])
                binary_y_train.append(-1)

        # Train a binary SVM classifier
        binary_classifier = SVC(kernel='linear')
        binary_classifier.fit(binary_X_train, binary_y_train)
        binary_classifiers.append(binary_classifier)

# Multi-class classification using SVM-2K
def svm_2k_predict(sample):
    confidence = [0] * num_classes

    for i, classifier in enumerate(binary_classifiers):
        class_i = i // (num_classes - 1)
        class_j = i % (num_classes - 1)

        if class_j >= class_i:
            class_j += 1

        prediction = classifier.predict([sample])

        if prediction == 1:
            confidence[class_i] += 1
        else:
            confidence[class_j] += 1

    return confidence.index(max(confidence))

# Make predictions for the test set
y_pred = [svm_2k_predict(sample) for sample in X_test]

# Evaluate the performance
svm2k_acc = accuracy_score(y_test, y_pred)
print(f'SVM-2K Accuracy Score: {svm2k_acc}')

# Confusion Matrix
svm2k_cm = confusion_matrix(y_test,y_pred)
print(f'SVM-2K Confusion Matrix: {svm2k_cm}')

# Plot SVM-2K Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(svm2k_cm,annot=True,fmt='d',cmap='Reds')
plt.xlabel('Preidicted Classes')
plt.ylabel('Actual Classes')
plt.title('SVM-2K Confusion Matrix')
plt.show()


#RF 
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate
print(f'Random Forest Training Score: {rf_clf.score(X_train, y_train)}')
print(f'Random Forest Testing Score: {rf_clf.score(X_test, y_test)}')

rf_y_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy Score: {rf_acc}')

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_y_pred)
print(f'Random Forest Confusion Matrix: \n{rf_cm}')

# Plot Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.title('Random Forest Confusion Matrix')
plt.show()


# Plot of Accuracy Score Comparison
models = ['SVM', 'DecisionTree', 'SVM-2K', 'Random Forest']
scores = [svm_acc, dt_acc, svm2k_acc, rf_acc]

plt.figure(figsize=(10,8))
plt.bar(models,scores,color=['#61f2da', '#f26b61', '#f5c895'])
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Score Comparison')
plt.ylim(0,1)

for i, score in enumerate(scores):
  plt.text(i, score, f'{score: .2f}', ha='center',va='bottom')

plt.show()
 

















