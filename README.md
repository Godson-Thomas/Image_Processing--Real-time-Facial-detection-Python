<img src="/home/supernova/Software/GIT_Folder/CV_FACE_VIDEO/images_xmls_videos/1.gif" width="400"  /> <br><br>

# COMPUTER VISION
Computer Vision, often abbreviated as CV, is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.Moreover Computer vision focuses on replicating parts of the complexity of the human vision system and enabling computers to identify and process objects in images and videos in the same way that humans do. <br>**Image processing**  is a method to perform some operations on an image, in order to get an enhanced image or to extract some useful information from it.<br>
Here we are going to detect facial features from a real time video by using **HAAR CASCADE** classifier.
# HAAR CASCADE
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.<br>
OpenCV already contains many pre-trained classifiers for face, eyes, smile etc..So we will be using one of the pre-trained classifier here.
# Library
OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision. In short it is a library used for Image Processing. It is mainly used to do all the operation related to Images.We will be using this library.
### Note : ***Video is basically a sequence of moving images."Persistance of vision" (speciality of our eyes)  plays a major role in receiving moving images.*** 
# Steps
## Installation :


1. We will be using Jupyter Notebook for writing the code.Make sure you have Jupyter Notebook installed.<br><br>
2. Lauch your Jupyter Notebook<br><br>
3. Now we have to install the OpenCV library.Type the code in the cell of Jupyter Notebook and run it.
```
pip install opencv-python
```
<br>
<img src="https://github.com/Godson-Thomas/Image_Processing---Facial-Detection-Using-OpenCV/blob/master/Images/2.png" width="500" height=75>  <br><br> 

4. - Download the ***Haar Cascade Classifiers***. [click here](https://raw.githubusercontent.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/master/cars.xml)<br>


 ## Code :
 ### Type the codes in the cell and run it.<br><br>
5. Import the OpenCV library and time module.
```
import cv2
import time
```
6. In OpenCV, a video can be read either by using the feed from a camera connected to a computer or by reading a video file. The first step towards reading a video file is to create a VideoCapture object. Its argument can be either the device index or the name of the video file to be read.<br>
In most cases, only one camera is connected to the system. So, all we do is pass ‘0’ and OpenCV uses the only camera attached to the computer. When more than one camera is connected to the computer, we can select the second camera by passing ‘1’, the third camera by passing ‘2’ and so on.
```
video=cv2.VideoCapture(0)

```
7. Now read the Haar Cascade classifiers.
```
face_cascade1=cv2.CascadeClassifier("/CV_FACE_VIDEO/haarcascade_frontalface_default.xml")
face_cascade2=cv2.CascadeClassifier("/CV_FACE_VIDEO/haarcascade_eye.xml")
face_cascade3=cv2.CascadeClassifier("/CV_FACE_VIDEO/haarcascade_mcs_mouth.xml")
face_cascade4=cv2.CascadeClassifier("/CV_FACE_VIDEO/haarcascade_mcs_nose.xml")
```
8. Since a video is a sequence of images, we have to go through each and every frame using a loop. We will use the specific haar cascade in every frame.Then we'll try to detect the eyes,mouth,nose and face in every frame.<br><br>
9. When these are detected,we'll draw a rectangle to indicate it.
```
a=1

while True:
    
    
    check,img = video.read(0)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces1 = face_cascade1.detectMultiScale(
        gray,

        scaleFactor=1.06,
        minNeighbors=4,                                         #Face
        minSize=(30, 30)
                )
    faces2 = face_cascade2.detectMultiScale(
        gray,

        scaleFactor=1.07,
        minNeighbors=5,                                         #Eyes
        minSize=(30, 30)
                )
    faces3 = face_cascade3.detectMultiScale(
        gray,

        scaleFactor=1.03,
        minNeighbors=4,                                         #Mouth
        minSize=(30, 30)
                )
    faces4 = face_cascade4.detectMultiScale(
        gray,

        scaleFactor=1.06,
        minNeighbors=3,                                         #Nose
        minSize=(30, 30)
                )

    for (x ,y ,w ,h) in faces1:
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(255 ,28 ,0) ,3)
    
    for (x ,y ,w ,h) in faces2:
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(255 ,255,0) ,3)
    for (x ,y ,w ,h) in faces3:
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(255 ,0 ,255) ,3)
    for (x ,y ,w ,h) in faces4:
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,( 0,0 ,255) ,3)
    
    cv2.imshow("Detected_Frame" ,img)

    k=cv2.waitKey(30)
    a=a+1
    if k==ord('q'):     # press 'q' to quit
        break

video.release()
cv2.destroyAllWindows()


```
<br>

10. Every frame is displayed on a window using :
```
 cv2.imshow("video" ,img)
 ```
 ### Note :
 Make sure that you destroy all the opened windows.
 ```
 video.release()
cv2.destroyAllWindows()

```
### Full Code :
[Click here](https://github.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/blob/master/Detection.ipynb)