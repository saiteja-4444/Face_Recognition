<h1 align="center"> FACE RECOGNISATION USING MACHINE LEARNING </h1>
<p align="center">

<a target="_blank" href="">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*x1JZIt-6lSDI5UPjP7Hdxw.jpeg" alt="Face" height="350px"/>
 </a>
  
</p>

# Table Of Contents
* [Abstract](#abstract)
* [Introduction](#introduction)
* [Objective](#objective)
* [Installation]()
* [Methods for Face Recognition]()
* [Applications of Face Recognition]()
* [Conclusion]()
* [Resources]()

#  ABSTRACT 

   Face recognition has gained significant traction as a reliable method for personal identification and authentication. This paper introduces an innovative approach to Face information recognition using artificial intelligence algorithms. We analyze the limitations of existing Face recognition systems, propose an advanced system leveraging AI techniques, discuss the modular components of the proposed system, and highlight the potential impact on security, convenience, and accuracy in various applications. By harnessing the power of AI, this approach presents a promising solution to enhance Face recognition systems. 

# INTRODUCTION  
In today's digital age, secure and efficient methods of personal identification and authentication are essential for safeguarding sensitive information, maintaining privacy, and enabling seamless interactions across various domains. Face recognition, which leverages unique physiological or behavioral characteristics of individuals, has emerged as a powerful solution in this endeavor. This paper delves into the innovative realm of Face information recognition, enhanced by the capabilities of artificial intelligence (AI) algorithms.

## Introduction To Artificial Intelligence 
-	In the early days of artificial intelligence, the field rapidly tackled and solved problems that are intellectually difficult for human beings but relatively straight-forward for computers problems that can be described by a list of formal, mathematical rules. The  true challenge to artificial intelligence proved to be solving the tasks that are easy for people to perform but hard for people to describe formally problems that we solve intuitively, that feel automatic, like recognizing spoken words or face sin images.
-	Artificial intelligence (AI) is an area of computer science that aims at emphasizing the ability of machines to behave the way humans do which includes complex tasks such as observing, learning, planning and making decisions for problem-solving.

![image](https://github.com/saiteja-4444/Face_Recognisation/assets/140083199/3fc396fc-d7de-476e-b40a-a3293a3b4cf3)


# OBJECTIVE
## PROPOSED SYSTEM 
We propose an advanced Face information recognition system that harnesses the capabilities of artificial intelligence algorithms. The proposed system comprises the following components:
### Face Data Acquisition :
- Collect diverse Face data, including fingerprints, iris patterns, facial images, and voice samples.
-  Ensure high-quality data collection to enhance accuracy.

### Feature Extraction and Representation :
- 	Utilize AI techniques to extract relevant features from Face data.
- 	Convert Face data into suitable representations for AI model input.
### Real-time Recognition :
- 	Develop real-time recognition algorithms capable of identifying individuals in real-world scenarios.
- 	Enable quick and accurate verification or identification.


# Installation


The model is built in Opencv 4.4.0.46 and tested on Windows 10,11 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
Python==3.7.4
pip install Flask
pip install numpy
pip install opencv_python
pip install pandas
pip install Pillow
pip install opencv-contrib-python
pip install matplotlib scikit-image opencv-python 
```


# Methods for Face Recognition
## LBPH Algorithm :
The Local Binary Patterns Histograms (LBPH) algorithm is a popular and robust method for face recognition. It extracts local texture information from facial images, encoding patterns and their relationships into a histogram representation. LBPH is computationally efficient and works well with facial images exhibiting variations in lighting conditions and facial expressions.

```
import cv2

face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with sample face images
def train_faces():
    faces = []
    labels = []
    # Load face images and corresponding labels
    # Preprocess the images, detect faces, and append them to the 'faces' list
    # Assign appropriate labels to each face and append them to the 'labels' list
    
    recognizer.train(faces, np.array(labels))

# Perform face recognition on a test image
def recognize_face(test_image):
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi_gray)
        # Perform further actions based on the predicted label and confidence level
```
## DLib Library for Face Recognition : 
DLib is a powerful open-source library that offers face detection, landmark estimation, and face recognition capabilities. It utilizes deep learning-based models, such as the ResNet network, to achieve high accuracy in face recognition tasks. The library provides pre-trained models for face recognition, simplifying the implementation process.

```
import dlib
import cv2
import numpy as np

# Load pre-trained models for face detection and facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('path/to/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('path/to/dlib_face_recognition_resnet_model_v1.dat')

# Load a sample image for face recognition
image = cv2.imread('path/to/sample_image.jpg')

# Detect faces in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

# Iterate over detected faces and perform face recognition
for face in faces:
    landmarks = predictor(gray, face)
    face_descriptor = facerec.compute_face_descriptor(image, landmarks)
    
    # Compare face descriptor with a known face database for recognition
    
    # Perform further actions based on the recognized identity
```

# Applications of Face Recognition 
Face recognition finds practical applications in various domains, including:
### Biometric Security :
Face recognition is used for secure authentication and access control systems, replacing traditional methods like passwords or ID cards.
### Surveillance and Law Enforcement :
Face recognition aids in identifying individuals in real-time or forensic scenarios, assisting law enforcement agencies in investigations.
### Attendance Management :
Face recognition technology automates attendance tracking in educational institutions or workplaces, eliminating the need for manual processes.
### Personalized Marketing ;
Retailers and advertisers use face recognition to analyze customer demographics and provide personalized experiences based on their profiles.
### Human-Computer Interaction :
Face recognition enables natural and intuitive interaction between humans and computers, allowing applications like emotion detection and virtual try-on.

# CONCLUSION 

Face information recognition using artificial intelligence algorithms represents a substantial advancement in security and identification technologies. The proposed system's integration of AI techniques enhances accuracy, robustness, and adaptability across various Face sources. By combining Face data acquisition, deep learning models, and fusion techniques, this approach demonstrates the potential to revolutionize the accuracy and efficiency of Face recognition systems. The project's modules together contribute to an effective, efficient, and reliable solution that aligns with evolving security requirements and technological advancements.



# Resources

* I got big help from [pyimagesearch](https://www.pyimagesearch.com/pyimagesearch-gurus/). It cleared most of my concepts with very Ease and practical examples.
* [Adam Geitgey](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) : How Face Recognition works
* [Face Recognition](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
* [Custom Dataset](https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/): helped me to create my own dataset.
* [Must Read Post](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)
* [Deep Face Recognition](http://krasserm.github.io/2018/02/07/deep-face-recognition/)
* [Facial Landmarks](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
* [Facial Alignment](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
* [Face Recognition for Beginners](https://towardsdatascience.com/face-recognition-for-beginners-a7a9bd5eb5c2)
* [Face Recognition Basics](https://www.coursera.org/lecture/convolutional-neural-networks/what-is-face-recognition-lUBYU)



