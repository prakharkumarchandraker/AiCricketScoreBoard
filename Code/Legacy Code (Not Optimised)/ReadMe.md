Frist We Install Anaconda Distribution From https://www.anaconda.com/products/distribution .

Make sure to give Admin Access while installing it. 

Your system must have Python Installed in it, if Not Please follow the steps below.

  Download python from here which ever is the latest version availableh ttps://www.python.org/downloads/
  
  Install and Give all admin rights.
  
  To check if Python is correctly installed, open command prompt and write "python" and press enter. This will show the version of Python Installed.
  
Now open the Anaconda Prompt (Anaconda 3)

Write cd <directory where code is present> and press enter. For example : cd C:\Users\prakh\Desktop\Cricket_Score

Write "Jupyter notebook" and press enter to open Jupyter Notebook.


Module 1 - This module is based on OpenCV. Here we capture image within green box in small size for fast processing.

Note: When running Module 1 ,You may encounter an error called "no cv2 Module found",this is because openCV is not installed. To solve this please follow the steps below.

  Open Anaconda Prompt as Admin and run the following scripts.
  
  conda update anaconda-navigator  
  
  conda update navigator-updater 
  
  pip install opencv-python
  

After All this you may want to change the location based on your preference of where to store images.

By default this code captures 1000 images in Jpg format of 50px X 50px.

A frame will appear with a green box. This green box will capure and crop and save photo.

We can change the size of green box in Line 16 #cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)

The 2nd parameter puts the green box in an area that is below 300 in x axis and 300 in y axist from top left.

The 3rd parameter decides the size of the green box.

The 4th Parameter decided the color of Green box.

If the size of the green box is changed we need to change code of line 21 as well with respected to the size of the green box #crop_img = frame[100:300, 100:300]

The frame will start capturing photos when key 'c' is pressed and will automatically stop after it has taken 1000 images.

We can manually stop the image capturing process by clicking the key 'q'


Module 2 - This module is based on OpenCV. Here we Preprocess the captured image which is saved in dataset folder.
  
First we need to create a folder named "Preprocessed" in the current directory and a folder named "Train" inside "Preprocessed" folder.

Code of Module 2 will take images from dataset folder and apply grayscale and convert it into binary image consisting of only black and white.
