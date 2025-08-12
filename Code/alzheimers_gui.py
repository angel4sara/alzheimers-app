# Importing libraries
import cv2
import time
import joblib
import numpy as np 
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog as fd
from PIL import Image,ImageTk
from customtkinter import *
from skimage.feature import hog
import warnings
warnings.filterwarnings('ignore')

#  Loading model for prediction
# Save the model

model = joblib.load('../SavedFiles/model.pkl')

model_type = type(model).__name__
print(f"Loaded Model: {model_type}")

# Creating window
root = Tk()
root.geometry('700x600')
root.resizable(0,0)
root.title('Alzheimer\'s')
root.configure(bg='#bcebf5')
Image_path = Image.open('../Extras/bg.jfif')
Image_path.putalpha(80)  
resize_image = Image_path.resize((700, 600))        #resizing img to fit the size of the window
image = ImageTk.PhotoImage(resize_image)
bgwin = Label(root, image=image)
bgwin.place(x=0,y=0) 


srcframe = Frame(root,width=700,height=600)
bgwin4 = Label(srcframe, image=image)
bgwin4.place(x=0,y=0) 

set_default_color_theme("green")        # setting button color as green

def select_image(path):     # Browse image and display image on window
    img_file = fd.askopenfilename(title='Select Image',
                                  filetype=[('Image files','*.jpeg;*.png')])
    path.delete(0,END)
    path.insert(END,img_file)
    if img_file:
        srcframe.pack(expand=True,fill=BOTH)

        image = Image.open(img_file)
        resize_image = image.resize((150, 190))
        photo = ImageTk.PhotoImage(resize_image)
        img.config(image=photo)
        img.image = photo


IMAGE_HEIGHT , IMAGE_WIDTH = 150, 150 # resize image 
def  input(image_path):     # preprocess and prediction of the input image
    inter.delete(1.0,END)
  
    start_time = time.time()
    inter.insert(END,f'\nThe image path - {image_path}')
    print(image_path)
   
    # Resizing and combining data
    roi_top_left = (50,50)
    roi_bottom_right = (100,100)

    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    test_img = cv2.imread(image_path)
    resized = cv2.resize(test_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # ROI extraction
    roi =  resized[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]   
    inter.insert(END,f'\n-------------------------------------------------------\nROI: {roi}')

    # Compute HOG features for the ROI
    hog_feature = hog(roi, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False,channel_axis=2)
    inter.insert(END,f'\n-------------------------------------------------------\nHOG: {hog_feature}')

    X_roi_reshaped = roi.reshape(1, -1)

    X_hog = hog_feature.reshape(1,hog_feature.shape[0])

    view = np.hstack(( X_hog,X_roi_reshaped))
    inter.insert(END,f'\n-------------------------------------------------------\nMulti view: {view}')

    prediction = model.predict(view)        # image prediction

    if prediction[0] == 0:
        inter.insert(END,f'\n-------------------------------------------------------\nResult is: "DEMENTED"')
        messagebox.showinfo('RESULT','DEMENTED')
        print('Demented')
    elif prediction[0] == 1:
        inter.insert(END,f'\n-------------------------------------------------------\nResult is: "MILD DEMENTED"')
        messagebox.showinfo('RESULT','MILD DEMENTED')

        print('Mild Demented')
    else:
        inter.insert(END,f'\n-------------------------------------------------------\nResult is: "NON DEMENTED"')
       
        messagebox.showinfo('RESULT','NON DEMENTED')

        print('Non Demented')


    elapsed_time = time.time() - start_time
    inter.insert(END,f'\nTotal time taken for execution {elapsed_time}secs')
  



def predict(image_path):  # Prediction 
    inter.delete(1.0,END)
 
    if path.get() != '':
        input(image_path)  # Perform prediction if a image selected
    else:
        messagebox.showwarning('Input','Please select a image')


def clearData():        # Clear texts and image
    path.delete(0,END)
    inter.delete(1.0,END)
    srcframe.pack_forget()

  
Label(root,text='Alzheimer\'s Detection',font=('Trobuchet',18,'bold'),fg='#800080',bg='#a7d0cd').place(x=200,y=20)

path = CTkEntry(root,width=300)
# path.place(x=120,y=70)
file = CTkButton(root,text='Browse',width=10,command=lambda: select_image(path))
file.place(x=25,y=250)

data = Label(root,text='Image',bg='#acdcd6').place(x=130,y=150)

img = Label(srcframe)
img.place(x=100,y=170)

Label(root,text='Process',bg='#bbe0e0').place(x=400,y=150)
inter = CTkTextbox(root,width=300,height=180)
inter.place(x=380,y=170)

submit = CTkButton(root,text='Predict',width=10,command=lambda: predict(path.get()))
submit.place(x=200,y=500)

clear = CTkButton(root,text='Clear',width=10,command=clearData)
clear.place(x=300,y=500)

root.mainloop()