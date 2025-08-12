# Importing libraries
import os
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

from skimage.feature import hog
from skimage.transform import resize


# Simple MKL

# Assigning height, width, train directory and class names
IMAGE_HEIGHT , IMAGE_WIDTH = 150, 150
CLASSES_LIST = ['AD','CI','CN']
train_dir = "../Dataset/mkl"

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

# Reshaping to 2D
X_roi_reshaped = X_roi.reshape(X_roi.shape[0], -1)

# Combining HOG and ROI features
multi_view_representation = np.hstack(( X_hog,X_roi_reshaped))
print('Multiview representation\n',multi_view_representation)

# Features
X = multi_view_representation

# Labels
le = LabelEncoder()
y = le.fit_transform(y)
print('Class Label Encoded',y)

#Train Test Splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Define kernel functions
def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def rbf_kernel(X1, X2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)

def poly_kernel(X1, X2, degree=2):
    return (np.dot(X1, X2.T) + 1) ** degree

kernels = [linear_kernel, rbf_kernel, poly_kernel]

# Create base kernel classifiers
base_classifiers = []

for kernel in kernels:
    # Compute kernel matrices
    X_train_kernel = kernel(X_train, X_train)

    # Create and train the base classifier with precomputed kernel
    clf = SVC(kernel='precomputed')
    clf.fit(X_train_kernel, y_train)

    # Append the trained classifier to the list
    base_classifiers.append(clf)

    del X_train_kernel
    gc.collect()

# Define parameter grid for SVM hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
}

# Perform grid search with cross-validation for each base classifier
tuned_classifiers = []
for kernel, clf in zip(kernels, base_classifiers):
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(kernel(X_train, X_train), y_train)
    tuned_classifiers.append(grid_search.best_estimator_)

    del grid_search
    gc.collect()

# Make predictions using the tuned classifiers

tuned_predictions = [clf.predict(kernel(X_test, X_train)) for kernel, clf in zip(kernels, tuned_classifiers)]


# Use majority voting to combine predictions
final_predictions = np.argmax(np.array(tuned_predictions), axis=0)

# Evaluate the final classifier
mkl_acc = (final_predictions == y_test).mean()
print(f'SimpleMKL Accuracy: {mkl_acc}')

mkl_cm = confusion_matrix(y_test,final_predictions)
print(f'Simple MKL Confusion Matrix: {mkl_cm}')

# Plot SVM Confusion Matrix
plt.figure(figsize=(10,6))
sns.heatmap(mkl_cm,annot=True,fmt='d',cmap='Reds')
plt.xlabel('Preidicted Classes')
plt.ylabel('Actual Classes')
plt.title('Simple MKL Confusion Matrix')
plt.show()