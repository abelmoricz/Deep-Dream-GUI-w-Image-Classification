import sys
from shutil import copyfile

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QScrollArea, QVBoxLayout, QGroupBox, QLabel, QPushButton, QFormLayout, QGridLayout, QSlider, QFileDialog, QProgressBar, QLineEdit
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import *
import PIL.Image, PIL.ImageDraw
import numpy as np
import tensorflow as tf

from deepdream import *
from classify import run_inference_on_image

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        super(QtWidgets.QWidget, self).__init__()

        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #variables to control operations of the class
        self.my_directory_path = str(os.path.dirname(os.path.realpath(__file__)))
        self.currentchannel = 29
        self.currentlayer = 'mixed4c_pool_reduce_pre_relu'
        self.noisepic = np.random.uniform(size=(193,193,3))  #193 fills up a 280x280 perfectly probably because of border thickness
        self.windows = list()
        self.nf_pic = PIL.Image.open('images/default')
        self.nf_pic = np.float32(self.nf_pic)
        self.import_pic = PIL.Image.open('images/default')
        self.import_pic = np.float32(self.import_pic)
        self.imported = False
        self.first_dream = False
        self.nf_save = False
        self.pic_save = False
        #variables for display purposes
        self.nf_num_iterations = 3
        self.pic_num_iterations = 5
        self.nf_classification = "No Description"
        self.pic_classification = "No Description"
        self.nf_layer = ""
        self.pic_layer = ""
        self.nf_channel = 0
        self.pic_channel = 0

        #displays options of Layer and Channel
        self.LC_pick = QtWidgets.QTabWidget(self.centralwidget)
        self.LC_pick.setGeometry(QtCore.QRect(10, 119, 380, 481))
        
        #Display selected layer and channel
        self.C_select = QtWidgets.QLabel(self.centralwidget)
        self.C_select.setGeometry(10, 60, 250, 25)
        self.C_select.setText("Channel: 30")
        self.L_select = QtWidgets.QLabel(self.centralwidget)
        self.L_select.setGeometry(10, 30, 250, 25)
        self.L_select.setText("Layer: mixed4c_pool_reduce")

        self.layer_array = ["conv2d0_pre_relu", "conv2d1_pre_relu","conv2d2_pre_relu","mixed3a_1x1_pre_relu","mixed3a_3x3_bottleneck_pre_relu","mixed3a_3x3_pre_relu","mixed3a_5x5_bottleneck_pre_relu","mixed3a_5x5_pre_relu","mixed3a_pool_reduce_pre_relu","mixed3b_1x1_pre_relu","mixed3b_3x3_bottleneck_pre_relu","mixed3b_3x3_pre_relu","mixed3b_5x5_bottleneck_pre_relu","mixed3b_5x5_pre_relu","mixed3b_pool_reduce_pre_relu","mixed4a_1x1_pre_relu","mixed4a_3x3_bottleneck_pre_relu","mixed4a_3x3_pre_relu","mixed4a_5x5_bottleneck_pre_relu","mixed4a_5x5_pre_relu","mixed4a_pool_reduce_pre_relu","mixed4b_1x1_pre_relu","mixed4b_3x3_bottleneck_pre_relu","mixed4b_3x3_pre_relu", "mixed4b_5x5_bottleneck_pre_relu","mixed4b_5x5_pre_relu","mixed4b_pool_reduce_pre_relu","mixed4c_1x1_pre_relu","mixed4c_3x3_bottleneck_pre_relu","mixed4c_3x3_pre_relu","mixed4c_5x5_bottleneck_pre_relu","mixed4c_5x5_pre_relu","mixed4c_pool_reduce_pre_relu","mixed4d_1x1_pre_relu","mixed4d_3x3_bottleneck_pre_relu","mixed4d_3x3_pre_relu","mixed4d_5x5_bottleneck_pre_relu","mixed4d_5x5_pre_relu","mixed4d_pool_reduce_pre_relu","mixed4e_1x1_pre_relu","mixed4e_3x3_bottleneck_pre_relu","mixed4e_3x3_pre_relu","mixed4e_5x5_bottleneck_pre_relu","mixed4e_5x5_pre_relu","mixed4e_pool_reduce_pre_relu"]
        self.channel_array = [64,64,192,64,96,128,16,32,32,128,128,192,32,96,64,192,96,204,16,48,64,160,112,224,24,64,64,128,128,256,24,64,64,112,144,288,32,64,64,256,160,320,32,128,128,256,160,320,48,128,128,384,192,384,48,128,128,128,128]


        #first loop creates the tabs each with the name of its Layer
        for layer in range(len(self.layer_array)):
            tab0 = QScrollArea()
            self.LC_pick.addTab(tab0, str(self.layer_array[layer][:-9]))
            picwidget = QWidget()
            tab0.setWidget(picwidget)
            tab0.setWidgetResizable(True)
            tab0.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            layout = QVBoxLayout()
            layout.addWidget(self.LC_pick)
            grid = QGridLayout(picwidget)
            #second loop creates the individual icons with their respective buttons
            rowcounter = 0 
            grid_x = 0
            grid_y = 0
            for i in range(self.channel_array[layer]):
                myPixmap = QtGui.QPixmap("thumbnails/" + self.layer_array[layer] + "/%s" %i)
                label = QtWidgets.QLabel(tab0)
                label.setPixmap(myPixmap)
                label.show()
                button = QtWidgets.QPushButton(str(i+1), tab0)
                button.setFixedWidth(58)
                button.clicked.connect(self.LC_select_func(str(i), layer))
                if rowcounter%4 == 0:
                    grid_x = grid_x + 2
                    grid_y = 0
                grid.addWidget(label,grid_x,grid_y)
                grid.addWidget(button,grid_x+1,grid_y)
                rowcounter = rowcounter + 1
                grid_y = grid_y + 1

        
                
        #Native Feature picture widget 
        self.NF_pic = QLabel(self.centralwidget)
        self.NF_pic.setGeometry(QtCore.QRect(410, 220, 380, 380))
        self.NF_pic.setFrameShape(QtWidgets.QLabel.StyledPanel)
        self.NF_pic.setFrameShadow(QtWidgets.QLabel.Raised)
        self.NF_status_bar = QProgressBar(self.NF_pic)
        self.NF_status_bar.setGeometry(93,183,200,20)
        
        #NF_info contains descrition, layer, channel, one button(pushtopic) and divider
        self.NF_info = QtWidgets.QWidget(self.centralwidget)
        self.NF_info.setGeometry(QtCore.QRect(600, 0, 200, 200))
        self.NF_description = QtWidgets.QLabel(self.NF_info)
        self.NF_description.setGeometry(QtCore.QRect(10, 70, 200, 100))
        self.NF_description.setAlignment(QtCore.Qt.AlignLeft)
        self.NF_LC = QtWidgets.QLabel(self.NF_info)
        self.NF_LC.setGeometry(QtCore.QRect(10, 10, 200, 50))
        self.NF_LC.setAlignment(QtCore.Qt.AlignLeft)
        self.NF_pushtopic = QtWidgets.QPushButton(self.NF_info)
        self.NF_pushtopic.setGeometry(QtCore.QRect(50, 164, 100, 25))
        self.NF_pushtopic.setText('Push to Pic')
        self.NF_pushtopic.clicked.connect(self.pushtopic_func)
        self.NF_divider = QtWidgets.QFrame(self.NF_info)
        self.NF_divider.setGeometry(QtCore.QRect(0, 10, 1, 190))
        self.NF_divider.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.NF_divider.setFrameShadow(QtWidgets.QFrame.Raised)
        
        #NF_controls contains four buttons, one slider and slider labels
        self.NF_controls = QtWidgets.QWidget(self.centralwidget)
        self.NF_controls.setGeometry(QtCore.QRect(400, 0, 200, 200))
        self.NF_window = QtWidgets.QPushButton(self.NF_controls)
        self.NF_window.setGeometry(QtCore.QRect(25, 162, 100, 25))
        self.NF_window.setText("New Window")
        self.NF_window.clicked.connect(self.NF_window_generate)
        self.NF_export = QtWidgets.QPushButton(self.NF_controls)
        self.NF_export.setGeometry(QtCore.QRect(25, 118, 100, 25))
        self.NF_export.setText("Save jpeg")
        self.NF_export.clicked.connect(self.NF_save_button_func)
        self.NF_classify = QtWidgets.QPushButton(self.NF_controls)
        self.NF_classify.setGeometry(QtCore.QRect(25, 75, 100, 25))
        self.NF_classify.setText("Classify")
        self.NF_classify.clicked.connect(self.NF_classify_pic_func)
        self.NF_dream = QtWidgets.QPushButton(self.NF_controls)
        self.NF_dream.setGeometry(QtCore.QRect(25, 12, 100, 45))
        self.NF_dream.setText("Dream")
        self.NF_dream.clicked.connect(self.NF_dream_func)
        self.NF_iter = QtWidgets.QSlider(self.NF_controls)
        self.NF_iter.setGeometry(QtCore.QRect(170, 40, 20, 141))
        self.NF_iter.setOrientation(QtCore.Qt.Vertical)
        self.NF_iter.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.NF_iter.setMaximum(10)
        self.NF_iter.setValue(3)
        self.NF_iter.valueChanged.connect(self.NF_iteration_change_func)

        #labels for the iterations slider
        self.NF_reps = QtWidgets.QLabel(self.NF_controls)
        self.NF_reps.setGeometry(QtCore.QRect(145, 10, 41, 17))
        self.NF_reps.setText("Iter.")
        self.NF_reps.setAlignment(QtCore.Qt.AlignCenter)        
        self.NF_0 = QtWidgets.QLabel(self.NF_controls)
        self.NF_0.setText("0")
        self.NF_0.setGeometry(QtCore.QRect(150, 165, 16, 17))
        self.NF_10 = QtWidgets.QLabel(self.NF_controls)
        self.NF_10.setText("10")
        self.NF_10.setGeometry(QtCore.QRect(150, 40, 16, 17))

        #picture widget for the result of a layer+channel imposed on a "normal" image aka Deep Dream
        self.PIC_pic = QtWidgets.QLabel(self.centralwidget)
        self.PIC_pic.setGeometry(QtCore.QRect(810, 220, 380, 380))
        self.PIC_pic.setFrameShape(QtWidgets.QLabel.StyledPanel)
        self.PIC_pic.setFrameShadow(QtWidgets.QLabel.Raised)
        self.PIC_status_bar = QProgressBar(self.PIC_pic)
        self.PIC_status_bar.setGeometry(93,183,200,20)

        #PIC_info contains descrition, layer, channel, two button "Import" and "Original" and asthetic divider
        #The "Original" button is added so that if the results of a deep dream aren't satisfying one doesn't have to go throught the import process again
        self.PIC_info = QtWidgets.QWidget(self.centralwidget)
        self.PIC_info.setGeometry(QtCore.QRect(1000, 0, 200, 200))
        self.PIC_description = QtWidgets.QLabel(self.PIC_info)
        self.PIC_description.setGeometry(QtCore.QRect(10, 70, 200, 100))
        self.PIC_description.setAlignment(QtCore.Qt.AlignLeft)
        self.PIC_LC = QtWidgets.QLabel(self.PIC_info)
        self.PIC_LC.setGeometry(QtCore.QRect(10, 10, 200, 50))
        self.PIC_LC.setAlignment(QtCore.Qt.AlignLeft)
        self.PIC_import = QtWidgets.QPushButton(self.PIC_info)
        self.PIC_import.setGeometry(QtCore.QRect(30, 164, 60, 25))
        self.PIC_import.setText("Import")
        self.PIC_import.clicked.connect(self.PIC_getfile)
        self.back_button = QtWidgets.QPushButton(self.PIC_info)
        self.back_button.setGeometry(QtCore.QRect(100, 164, 60, 25))
        self.back_button.setText("Original")
        self.back_button.clicked.connect(self.back_button_func)

        #NF_controls contains five buttons, one slider and slider labels
        self.PIC_controls = QtWidgets.QWidget(self.centralwidget)
        self.PIC_controls.setGeometry(QtCore.QRect(800, 0, 200, 200))
        self.PIC_window = QtWidgets.QPushButton(self.PIC_controls)
        self.PIC_window.setGeometry(QtCore.QRect(25, 162, 100, 25))
        self.PIC_window.clicked.connect(self.PIC_window_generate)
        self.PIC_window.setText("Original Size")
        self.PIC_export = QtWidgets.QPushButton(self.PIC_controls)
        self.PIC_export.setGeometry(QtCore.QRect(25, 118, 100, 25))
        self.PIC_export.setText("Save jpeg")
        self.PIC_export.clicked.connect(self.PIC_save_button_func)
        self.PIC_classify = QtWidgets.QPushButton(self.PIC_controls)
        self.PIC_classify.setGeometry(QtCore.QRect(25, 75, 100, 25))
        self.PIC_classify.setText("Classify")
        self.PIC_classify.clicked.connect(self.PIC_classify_pic_func)
        self.PIC_dream = QtWidgets.QPushButton(self.PIC_controls)
        self.PIC_dream.setGeometry(QtCore.QRect(25, 12, 100, 45))
        self.PIC_dream.setText("Dream")
        self.PIC_dream.clicked.connect(self.PIC_dream_func)
        self.PIC_iter = QtWidgets.QSlider(self.PIC_controls)
        self.PIC_iter.setGeometry(QtCore.QRect(170, 40, 20, 141))
        self.PIC_iter.setOrientation(QtCore.Qt.Vertical)
        self.PIC_iter.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.PIC_iter.setMaximum(10)
        self.PIC_iter.setValue(5)
        self.PIC_iter.valueChanged.connect(self.PIC_iteration_change_func)
        
        #labels for the iterations slider
        self.PIC_reps = QtWidgets.QLabel(self.PIC_controls)
        self.PIC_reps.setGeometry(QtCore.QRect(145, 10, 41, 17))
        self.PIC_reps.setText("Iter.")
        self.PIC_reps.setAlignment(QtCore.Qt.AlignCenter)
        self.PIC_0 = QtWidgets.QLabel(self.PIC_controls)
        self.PIC_0.setGeometry(QtCore.QRect(150, 165, 16, 17))
        self.PIC_0.setText("0")
        self.PIC_0.setAlignment(QtCore.Qt.AlignCenter)
        self.PIC_10 = QtWidgets.QLabel(self.PIC_controls)
        self.PIC_10.setGeometry(QtCore.QRect(150, 40, 16, 17))
        self.PIC_10.setText("10")
        self.PIC_10.setAlignment(QtCore.Qt.AlignCenter)

        #picture titles and divider
        self.NF_title = QtWidgets.QLabel(self.centralwidget)
        self.NF_title.setGeometry(QtCore.QRect(535, 200, 140, 25))
        self.NF_title.setAlignment(QtCore.Qt.AlignCenter)
        self.NF_title.setText("Native Feature")
        self.PIC_title = QtWidgets.QLabel(self.centralwidget)
        self.PIC_title.setGeometry(QtCore.QRect(935, 200, 140, 25))
        self.PIC_title.setAlignment(QtCore.Qt.AlignCenter)
        self.PIC_title.setText("Picture")
        self.PIC_divider = QtWidgets.QFrame(self.PIC_info)
        self.PIC_divider.setGeometry(QtCore.QRect(0, 10, 1, 190))
        self.PIC_divider.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PIC_divider.setFrameShadow(QtWidgets.QFrame.Raised)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    #functions taken directly from bapoczos script and modified to add progress bar functionality
    def showarray(self, a, name, fmt='jpeg'):
        '''create a jpeg file from an array'''
        # clip the values to be between 0 and 255
        a = np.uint8(np.clip(a, 0, 1)*255)
        PIL.Image.fromarray(a).save(name, fmt)
        
    def render_lapnorm(self, t_obj, img0=np.random.uniform(size=(193,193,3)), iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4, name="images/default"):
        self.NF_status_bar.show()
        if iter_n > 0:
            percent_added = (100/(iter_n*octave_n))
        else:
            percent_added = 100
        progress = 0

        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n)) # build the laplacian normalization graph
        img = img0.copy()
        for octave in range(octave_n):
            if octave>0:
                hw = np.float32(img.shape[:2])*octave_scale
                img = resize(img, np.int32(hw))
            for i in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                g = lap_norm_func(g)
                img += g*step
                progress = progress + percent_added
                self.NF_status_bar.setValue(progress)
            clear_output()
        self.NF_status_bar.hide()
        self.showarray((img-img.mean())/max(img.std(), 0.0001)*0.1 + 0.5, name)

    def render_deepdream(self,t_obj, img0=np.random.uniform(size=(193,193,3)), iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, name="images/render_deepdream_defaultname"):

        self.PIC_status_bar.show()
        if iter_n > 0:
            percent_added = (100/(iter_n*octave_n))
        else:
            percent_added = 100
        progress = 0
        
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

        # split the image into a number of octaves getting smaller and smaller images
        img = img0
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[:2] #image height and width
            lo = resize(img, np.int32(np.float32(hw)/octave_scale)) #low frequency parts (smaller image)
            hi = img-resize(lo, hw) #high frequency parts (details)
            img = lo # next iteration rescale this one
            octaves.append(hi) # add the details to octaves
        
        # generate details octave by octave from samll image to large
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for i in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
                clear_output()
                progress = progress + percent_added
                self.PIC_status_bar.setValue(progress)

                self.showarray(img/255.0, name)
        self.PIC_status_bar.hide()

    #design to allow a user to dream on top of a Native Feature
    def pushtopic_func(self):
        if self.first_dream:
            copyfile(self.my_directory_path + '/images/default', self.my_directory_path + '/images/import_raw')
            copyfile(self.my_directory_path + '/images/default', self.my_directory_path + '/images/import_dream')
            pixmap = QtGui.QPixmap("images/default")
            self.PIC_pic.setPixmap(pixmap)

            print("pushpick is exectuting")
            self.PIC_status_bar.hide()
            self.import_pic = PIL.Image.open('images/default')
            self.import_pic = np.float32(self.import_pic)
            self.imported = True
            self.pic_channel = int(self.nf_channel)
            self.pic_layer = str(self.nf_layer)
        else:
            self.NF_LC.setText("Dream a picture first")
    
    #imports a jpg or jpeg file
    def PIC_getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.my_directory_path ,"Image files (*.jpg *jpeg)") #opens dialog box
        copyfile(fname[0], self.my_directory_path + '/images/import_raw') #creates import_raw picture from import

        resized = PIL.Image.open(self.my_directory_path + '/images/import_raw')
        copyfile(self.my_directory_path + '/images/import_raw', self.my_directory_path + '/images/import_dream') #on first import sets import_dream picture equal to import_raw so you can classify it
        
        img_w, img_l = resized.size

        print(resized.size)

        if img_w > img_l:
            resized_length = int((img_l/img_w)*386)
            resized = resized.resize((386,resized_length), PIL.Image.ANTIALIAS)
        if img_w <= img_l:
            resized_width = int((img_w/img_l)*386)
            resized = resized.resize((resized_width,386), PIL.Image.ANTIALIAS)

        resized.save('images/resized_import_raw.jpeg')

        pixmap = QtGui.QPixmap('images/resized_import_raw.jpeg')
        self.PIC_pic.setPixmap(pixmap)
        self.PIC_status_bar.hide()

        self.import_pic = PIL.Image.open(fname[0])
        self.import_pic = np.float32(self.import_pic)
        self.imported = True
        self.PIC_LC.setText("")

    #selects the Layer and Channel from the menu
    def LC_select_func(self, channel, layer):
        def setvalues():
            self.currentchannel = channel
            displaychannel = str(int(channel)+1)
            self.C_select.setText("Channel: " + displaychannel)
            self.currentlayer = self.layer_array[layer]
            self.L_select.setText("Layer: " + self.layer_array[layer][:-9])
        return setvalues

    #runs "Simple image classification with Inception" from the Tensorflow Authors 
    def NF_classify_pic_func(self):
        if self.first_dream:
            classification = run_inference_on_image(self.my_directory_path + '/images/default')
            self.nf_classification = classification
            self.NF_description.setText(classification)
        else:
            self.NF_LC.setText("Dream a picture first")

    def PIC_classify_pic_func(self):
        if self.imported:
            classification = run_inference_on_image(self.my_directory_path + '/images/import_dream')
            self.pic_classification = classification
            self.PIC_description.setText(classification)
        else:
            self.PIC_LC.setText("Import a picture first")

    #If the user wants to easily compare the results of two different layers and channels they can do so by opening a window of the current result and experimenting further
    def NF_window_generate(self):
        if self.first_dream:
            new_window = picWindow(self)
            self.windows.append(new_window)
            new_window.show()
        else:
            self.NF_LC.setText("Dream a picture first")

    #shows the picture with dream effect in its original size
    def PIC_window_generate(self):
        if self.imported:
            new_window = import_picWindow(self)
            self.windows.append(new_window)
            new_window.show()
        else:
            self.PIC_LC.setText("Import a picture first")
    
    #creates an instance of the savePicture class 
    def NF_save_button_func(self):
        if self.first_dream:
            self.nf_save = True
            self.pic_save = False
            new_window = savePicture(self)
            self.windows.append(new_window)
            new_window.show()
        else:
            self.NF_LC.setText("Dream a picture first")

    def PIC_save_button_func(self):
        if self.imported:
            self.pic_save = True
            self.nf_save = False
            new_window = savePicture(self)
            self.windows.append(new_window)
            new_window.show()
        else:
            self.PIC_LC.setText("Import a picture first")

    #records changes in the number of iterations
    def NF_iteration_change_func(self):
        self.nf_num_iterations = self.NF_iter.value()

    def PIC_iteration_change_func(self):
        self.pic_num_iterations = self.PIC_iter.value()

    #creates the PIC deep dream result
    def PIC_dream_func(self):
        if self.imported:
            self.PIC_iteration_change_func()
            channel = int(self.currentchannel)
            displaychannel = channel + 1
            self.PIC_LC.setText("L: " + self.currentlayer[:-9] + "\nC: " + str(displaychannel) + "\nI: " + str(self.pic_num_iterations))
            self.PIC_description.setText("No Description")
            self.render_deepdream(T(self.currentlayer)[:,:,:,int(self.currentchannel)],name="images/import_dream", img0=self.import_pic, iter_n=self.pic_num_iterations)     
            self.import_pic = PIL.Image.open('images/import_dream')
            self.import_pic = np.float32(self.import_pic)
            resized = PIL.Image.open(self.my_directory_path + '/images/import_dream')
            img_w, img_l = resized.size
            if img_w > img_l:
                resized_length = int((img_l/img_w)*386)
                resized = resized.resize((386,resized_length), PIL.Image.ANTIALIAS)
            if img_w <= img_l:
                resized_width = int((img_w/img_l)*386)
                resized = resized.resize((resized_width,386), PIL.Image.ANTIALIAS)
            resized.save('images/resized_import_dream.jpeg')
            pixmap = QtGui.QPixmap('images/resized_import_dream.jpeg')
            self.PIC_pic.setPixmap(pixmap)
            self.pic_classification = "No Description"
            self.pic_layer = self.currentlayer
            self.pic_channel = self.currentchannel
        else:
            self.PIC_LC.setText("Import a picture first")

    #creates the Native Feature deep dream result
    def NF_dream_func(self):
        channel = int(self.currentchannel)
        displaychannel = channel + 1
        self.NF_LC.setText("L: " + self.currentlayer[:-9] + "\nC: " + str(displaychannel) + "\n I: " + str(self.nf_num_iterations))
        self.NF_description.setText("No Description")
        self.nf_classification = "No Description"
        self.render_lapnorm(T(self.currentlayer)[:,:,:,channel],img0=self.noisepic, iter_n = self.nf_num_iterations)
        self.nf_pic = PIL.Image.open('images/default')
        self.nf_pic = np.float32(self.nf_pic)
        self.nf_layer = self.currentlayer
        self.nf_channel = self.currentchannel
        pixmap = QtGui.QPixmap("images/default")
        self.NF_pic.setPixmap(pixmap)
        self.first_dream = True

    #resets to the original imported picture
    def back_button_func(self):
        if self.imported:
            copyfile(self.my_directory_path + '/images/import_raw', self.my_directory_path + '/images/import_dream')
            self.import_pic = PIL.Image.open('images/import_raw')
            self.import_pic = np.float32(self.import_pic)
            resized = PIL.Image.open(self.my_directory_path + '/images/import_raw')
            img_w, img_l = resized.size
            if img_w > img_l:
                resized_length = int((img_l/img_w)*386)
                resized = resized.resize((386,resized_length), PIL.Image.ANTIALIAS)
            if img_w <= img_l:
                resized_width = int((img_w/img_l)*386)
                resized = resized.resize((resized_width,386), PIL.Image.ANTIALIAS)
            resized.save('images/resized_import_raw.jpeg')
            pixmap = QtGui.QPixmap('images/resized_import_raw.jpeg')
            self.PIC_pic.setPixmap(pixmap)
            self.pic_classification = "No Description"
            self.pic_layer = ""
            self.pic_channel = 0
            self.pic_num_iterations = 0
            self.PIC_LC.setText("L: "  + "\nC: " +  "\n I: ")
            self.PIC_description.setText("")
        else:
            self.PIC_LC.setText("Import a picture first")

#creates a smaller summary of an individual picture including: picture, layer, channel, number of iterations, description, and "Save" button
class picWindow(Ui_MainWindow):
    def __init__(self, Ui_MainWindow):
        QWidget.__init__(self, None)
        self.lay = Ui_MainWindow.nf_layer
        self.ch = int(Ui_MainWindow.nf_channel)
        self.des = Ui_MainWindow.nf_classification
        self.iter = int(Ui_MainWindow.nf_num_iterations)
        self.save_func = Ui_MainWindow.NF_save_button_func
        self.initUI()

    def initUI(self):
        self.pic = QtWidgets.QLabel(self)
        self.pic.setGeometry(0,0,380,380)
        pixmap = QtGui.QPixmap("images/default")
        self.pic.setPixmap(pixmap)
        self.layer_channel = QtWidgets.QLabel(self)
        self.layer_channel.setGeometry(390,0,250,380)
        displaychannel = self.ch + 1
        text = "Layer:\n" + self.lay[:-9] + "\n" + "\nChannel:\n" + str(displaychannel) +"\n\nIterations:\n" + str(self.iter) + "\n\nDescription:\n" + self.des
        self.layer_channel.setText(text)

#same as picWindow except that it displays the modified dream image in its original size
class import_picWindow(Ui_MainWindow):
    def __init__(self,Ui_MainWindow):
        QWidget.__init__(self,None)
        self.lay = Ui_MainWindow.pic_layer
        self.ch = int(Ui_MainWindow.pic_channel)
        self.des = Ui_MainWindow.pic_classification
        self.iter = int(Ui_MainWindow.pic_num_iterations)
        self.save_func = Ui_MainWindow.PIC_save_button_func
        self.my_directory_path = Ui_MainWindow.my_directory_path
        self.initUI()

    def initUI(self):
        size = PIL.Image.open(self.my_directory_path + '/images/import_dream')
        img_w, img_l = size.size

        self.pic = QtWidgets.QLabel(self)
        self.pic.setGeometry(0,0,img_w,img_l)
        pixmap = QtGui.QPixmap("images/import_dream")
        self.pic.setPixmap(pixmap)

        self.layer_channel = QtWidgets.QLabel(self)
        self.layer_channel.setGeometry(img_w + 10,0,250,380)
        displaychannel = self.ch + 1
        text = "Layer:\n" + self.lay[:-9] + "\n" + "\nChannel:\n" + str(displaychannel) +"\n\nIterations:\n" + str(self.iter) + "\n\nDescription:\n" + self.des
        self.layer_channel.setText(text)

#the new window generated responsible for saving the image to jpeg
class savePicture(Ui_MainWindow):
    def __init__(self, Ui_MainWindow):
        QWidget.__init__(self,None)
        self.name = "no_name"
        self.NF_pic = Ui_MainWindow.nf_pic
        self.PIC_pic = Ui_MainWindow.import_pic
        self.nf_save = Ui_MainWindow.nf_save
        self.pic_save = Ui_MainWindow.pic_save
        self.my_directory_path = Ui_MainWindow.my_directory_path
        self.initUI()
    
    def initUI(self):
        self.name_input = QLineEdit(self)
        self.name_input.setGeometry(10,10,180,30)
        self.name_input.setText("no_name")
        self.save_button = QPushButton(self)
        self.save_button.setText("Save")
        self.save_button.setGeometry(200,10,50,30)
        self.save_button.clicked.connect(self.save_func)

    def save_func(self):
        if self.nf_save:
            filename = "default"
        if self.pic_save:
            filename =  "import_dream"
        name = self.name_input.text()
        copyfile(self.my_directory_path + '/images/%s'%filename, self.my_directory_path + '/images/%s'%name)
        self.name_input.setText("Sucessfully Saved")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
