# Deep Dream GUI with image classification
First, credit goes to Github user bapoczos and the Tensorflow team for providing the code for deepdream.py and classify.py  

This is my first public project; criticism is welcomed. 
The focus was usability and intuitiveness.

Features:
* Creates Native features; a quick visualization of how a layer and channel are going to distort an image
* Standard Deep Dream algorithm imposed on a jpeg
* Inception classification of created images
* Visual menu for layers and 5049 different channels  
* Ability to compare different dreams without losing the information to recreate any of those dreams  
(because I was tired of choosing layers and channels randomly and waiting each time for what would result)  
 
    
Requirements:  
Python 3.6.8 (try different versions at your own risk) 
  
  
PyQt5```pip3 install PyQt5```  
  
  
Pillow```pip3 install Pillow```  
  
  
numpy```pip3 install numpy```  
  
  
tensorflow```python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl```  
  
  
IPython```pip3 install ipython```

**Unzip the thumbnails folder before running GUI.py**

![alt text](https://github.com/abelmoricz/Deep-Dream-w-Image-Classification/blob/master/GUI.png "Logo Title Text 1")




