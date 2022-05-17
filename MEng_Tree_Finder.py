import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import csv
import random
import cv2
import sys
import os
from PIL import Image
from torch.autograd import Variable
from canny_edge_detector import cannyEdgeDetector
from skimage.measure import approximate_polygon, find_contours
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from bingapi import bing_getaerial, find_postcode
from PyQt5 import QtCore, QtGui, QtWidgets


# This class is for the GUI functions.
# Change the file locations to locations on your computer to enable the prorgam to work
#Change the Bing API key in the BingAPI file.

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1123, 798)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setGeometry(QtCore.QRect(390, 10, 351, 351))
        self.Image.setText("")
        self.Image.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/testmap.png"))
        self.Image.setScaledContents(True)
        self.Image.setObjectName("Image")
        self.mask = QtWidgets.QLabel(self.centralwidget)
        self.mask.setGeometry(QtCore.QRect(760, 10, 351, 351))
        self.mask.setText("")
        self.mask.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/mask.png"))
        self.mask.setScaledContents(True)
        self.mask.setObjectName("mask")
        self.edge = QtWidgets.QLabel(self.centralwidget)
        self.edge.setGeometry(QtCore.QRect(760, 380, 351, 351))
        self.edge.setText("")
        self.edge.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/edge.png"))
        self.edge.setScaledContents(True)
        self.edge.setObjectName("edge")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(400, 20, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(770, 20, 31, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(770, 390, 31, 16))
        self.label_3.setObjectName("label_3")
        self.tree_height_mask = QtWidgets.QLabel(self.centralwidget)
        self.tree_height_mask.setGeometry(QtCore.QRect(390, 380, 351, 351))
        self.tree_height_mask.setText("")
        self.tree_height_mask.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/height.png"))
        self.tree_height_mask.setScaledContents(True)
        self.tree_height_mask.setObjectName("tree_height_mask")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 390, 60, 16))
        self.label_4.setObjectName("label_4")
        self.red_tree = QtWidgets.QLabel(self.centralwidget)
        self.red_tree.setGeometry(QtCore.QRect(30, 580, 2000, 50))
        self.red_tree.setObjectName("red_tree")
        self.yellow_tree = QtWidgets.QLabel(self.centralwidget)
        self.yellow_tree.setGeometry(QtCore.QRect(30, 620, 2000, 50))
        self.yellow_tree.setObjectName("yellow_tree")
        self.blue_tree = QtWidgets.QLabel(self.centralwidget)
        self.blue_tree.setGeometry(QtCore.QRect(30, 660, 2000, 50))
        self.blue_tree.setObjectName("blue_tree")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 371, 121))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.postcode_text = QtWidgets.QLabel(self.frame)
        self.postcode_text.setGeometry(QtCore.QRect(20, 80, 321, 31))
        self.postcode_text.setObjectName("postcode_text")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(130, 50, 87, 20))
        self.checkBox.setObjectName("checkBox")
        self.CoordsLabel = QtWidgets.QLabel(self.frame)
        self.CoordsLabel.setGeometry(QtCore.QRect(10, 20, 121, 16))
        self.CoordsLabel.setObjectName("CoordsLabel")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(210, 10, 164, 32))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.Coords = QtWidgets.QLineEdit(self.frame)
        self.Coords.setGeometry(QtCore.QRect(130, 10, 161, 31))
        self.Coords.setObjectName("Coords")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 540, 371, 181))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.number_of_trees = QtWidgets.QLabel(self.frame_2)
        self.number_of_trees.setGeometry(QtCore.QRect(20, 16, 500, 16))
        self.number_of_trees.setObjectName("number_of_trees")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 140, 371, 391))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.image1 = QtWidgets.QLabel(self.frame_3)
        self.image1.setGeometry(QtCore.QRect(10, 20, 331, 31))
        self.image1.setWordWrap(True)
        self.image1.setObjectName("image1")
        self.mask1 = QtWidgets.QLabel(self.frame_3)
        self.mask1.setGeometry(QtCore.QRect(10, 100, 301, 16))
        self.mask1.setObjectName("mask1")
        self.height2 = QtWidgets.QLabel(self.frame_3)
        self.height2.setGeometry(QtCore.QRect(10, 150, 341, 51))
        self.height2.setWordWrap(True)
        self.height2.setObjectName("height2")
        self.mask2 = QtWidgets.QLabel(self.frame_3)
        self.mask2.setGeometry(QtCore.QRect(10, 230, 341, 31))
        self.mask2.setWordWrap(True)
        self.mask2.setObjectName("mask2")
        self.area_red = QtWidgets.QLabel(self.frame_3)
        self.area_red.setGeometry(QtCore.QRect(10, 280, 500, 16))
        self.area_red.setObjectName("area_red")
        self.area_yellow = QtWidgets.QLabel(self.frame_3)
        self.area_yellow.setGeometry(QtCore.QRect(10, 320, 500, 16))
        self.area_yellow.setObjectName("area_yellow")
        self.are_blue = QtWidgets.QLabel(self.frame_3)
        self.are_blue.setGeometry(QtCore.QRect(10, 360, 500, 16))
        self.are_blue.setObjectName("are_blue")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1123, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.buttonBox.clicked.connect(self.on_click)

    #This function runs when the search button is clicked
    def on_click(self): 
        coords_text= self.Coords.text()
        print(coords_text)
        if self.checkBox.isChecked() == True:
            coords_text = find_postcode(coords_text)
        tree_height = find_trees(coords_text)
        self.Image.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/testmap.png"))
        self.mask.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/mask.png"))
        self.edge.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/edge.png"))
        self.tree_height_mask.setPixmap(QtGui.QPixmap("/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/height.png"))
        self.red_tree.setText("Red Tree = "+str(tree_height[0][0])+'m (max ='+str(tree_height[1][0])+'m, min ='+str(tree_height[2][0])+'m)')
        self.yellow_tree.setText("Green Tree = "+str(tree_height[0][1])+'m (max ='+str(tree_height[1][1])+'m, min ='+str(tree_height[2][1])+'m)')
        self.blue_tree.setText("Blue Tree = "+str(tree_height[0][2])+'m (max ='+str(tree_height[1][2])+'m, min ='+str(tree_height[2][2])+'m)')
        red_area = (tree_height[4][0])
        green_area = (tree_height[4][1])
        blue_area = (tree_height[4][2])
        self.number_of_trees.setText("Number of Detected Trees ="+ str(tree_height[3]))
        self.area_red.setText("Red Tree Area = "+"{0:.{1}f} m\u00b2".format(red_area, 2)) #{0:.{1}f} m\u00b2 is to generate the squred symbol.
        self.area_yellow.setText("Green Tree Area = "+"{0:.{1}f} m\u00b2".format(green_area, 2))
        self.are_blue.setText("Blue Tree Area = "+"{0:.{1}f} m\u00b2".format(blue_area, 2))


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MEng Tree Finder"))
        self.label.setText(_translate("MainWindow", "Image"))
        self.label_2.setText(_translate("MainWindow", "Mask"))
        self.label_3.setText(_translate("MainWindow", "Edge"))
        self.label_4.setText(_translate("MainWindow", "Height"))
        self.label.setStyleSheet("QLabel { color : white; }");
        self.label_2.setStyleSheet("QLabel { color : white; }");
        self.label_3.setStyleSheet("QLabel { color : white; }");
        self.label_4.setStyleSheet("QLabel { color : white; }");
        self.red_tree.setText(_translate("MainWindow", "Red Tree = "))
        self.red_tree.setStyleSheet("QLabel { color : red; }");
        self.yellow_tree.setText(_translate("MainWindow", "Green Tree ="))
        self.yellow_tree.setStyleSheet("QLabel { red; color : green; }");
        self.blue_tree.setText(_translate("MainWindow", "Blue Tree ="))
        self.blue_tree.setStyleSheet("QLabel { color : blue; }");
        self.postcode_text.setText(_translate("MainWindow", "Check the Postcode Box when entering a Postcode"))
        self.checkBox.setText(_translate("MainWindow", "Postcode"))
        self.CoordsLabel.setText(_translate("MainWindow", "Enter Coordinates:"))
        self.image1.setText(_translate("MainWindow", "Image: Gathered from Microsoft Bing Maps."))
        self.mask1.setText(_translate("MainWindow", "Mask: Mask of all trees identified in the image."))
        self.height2.setText(_translate("MainWindow", "Height: Three trees chosen at random with an estimation of their heights."))
        self.mask2.setText(_translate("MainWindow", "Edge: The edge of the masks has been detected using a Canny Edge Detector"))
        self.area_red.setText(_translate("MainWindow", "Red Tree Area = "))
        self.area_yellow.setText(_translate("MainWindow", "Green Tree Area = "))
        self.are_blue.setText(_translate("MainWindow", "Blue Tree Area = "))
        self.area_red.setStyleSheet("QLabel { color : red; }");
        self.area_yellow.setStyleSheet("QLabel { color : green; }");
        self.are_blue.setStyleSheet("QLabel { color : blue }");
        self.number_of_trees.setText(_translate("MainWindow", "Number of Detected Trees = "))

# This is the main function of the progrm - Could be split into smaller functions if used in further work
def find_trees(center_latlon):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        loader = transforms.Compose([transforms.ToTensor()])

        # get map image
        if not os.path.exists('./mapsamples'):
            os.makedirs('./mapsamples')
        bing_getaerial(center_latlon)
        img = Image.open('/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/testmap.png')
        img = loader(img.convert('RGB'))
        num_classes = 2

        # load model from a trained network
        model = get_model_instance_segementation(num_classes)
        model.load_state_dict(torch.load('/Users/Hadley 1/Documents/MEng Project/python/local/model_weights_ResNet50_COCO.pth', map_location=device))
        model.eval()
        model.to(device)
        with torch.no_grad():
            prediction = model([img.to(device)])


        img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        num_obj = len(prediction[0]['masks'][:, 0])

        pic = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
        ret, pic1 = cv2.threshold(pic, 200, 255, cv2.THRESH_BINARY)
        output_array = pic1

        # Go though each of the masks that is produced and threshold each of them to make the mask appear sharper.
        # Add each of the masks to the output array.
        for i in range(1,len(prediction[0]['masks'])):
          pic = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
          ret, pic1 = cv2.threshold(pic, 200, 255, cv2.THRESH_BINARY)
          output_array = output_array + pic1

        img3 = Image.fromarray(output_array)
        img_t = np.array(img3)
        
        Image.fromarray(output_array)

        image_array = []
        area_array =[]
        colour_array =[]
        #Count area of each of the masks
        #This is done by counting the nuber of pixels that are white (255)
        for i in range(len(prediction[0]['masks'])):
          pic = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
          ret, pic1 = cv2.threshold(pic, 200, 255, cv2.THRESH_BINARY)
          occurrences = pic1 == 255
          total = occurrences.sum()
          area_array.append(total)
          output_pic = Image.fromarray(pic1)
          output_pic_p = output_pic.convert("RGB") # Convert to an RGB image - adds 3 channels
          image_array.append(output_pic_p)


        width = 600 
        height = 600
        # Randomly generate 3 values R,G,B to make a colour for each of the masks
        # Set each of the masks to th generated colour.
        for k in range (len(prediction[0]['masks'])):
          img = image_array[k]
          R = random.randint(20,255)
          G = random.randint(20,255)
          B = random.randint(20,255)
          colour_array.append((R,G,B))
          for i in range(0,width):# process all pixels
              for j in range(0,height):
                  data = img.getpixel((i,j))
                  if (data[0]==255 and data[1]==255 and data[2]==255):
                      img.putpixel((i,j),(R,G,B))


        # Go thorough all of the masks that have been generated and see if there is already a tree at thaat lcoation
        # If there is a tree add the tree that has the largest area.
        # If a tree that has a smaller area is already there, remove it and add the larger one.

        final_image = np.array(image_array[0])
        number_of_trees_in_image =0
        for c in range(len(image_array)):
          add_image = True
          img = image_array[c]
          img_array_format = np.array(img)
          for i in range(0,width):# process all pixels
              for j in range(0,height):
                  data = img_array_format[i,j]
                  data_final = final_image[i,j]
                  if (data[0]!=0):
                    #print(str(data[0])+' '+str(data_final[0]))
                    if(data_final[0]!=0):
                      add_image = False
                      print("NO ADD")
                      break
              else:
                  continue  
              break   
          
          if (add_image == True):
            final_image += np.array(image_array[c])
            number_of_trees_in_image +=1
        
        imgx1_rgb = Image.fromarray(final_image).convert('RGB')


        # Generate the outline of the mask 
        for idx in range(1, num_obj+1):
            img3 = Image.fromarray(prediction[0]['masks'][idx-1, 0].mul(255).byte().cpu().numpy())
            img_t = np.array(img3)
            ret, img3 = cv2.threshold(img_t, 200, 255, cv2.THRESH_TOZERO)
            imgx = img3
            img3 = loader(img3).unsqueeze(0)
            img3.to(device, torch.float)
            grad = gradient_img(img3)
            grad = grad.detach().cpu().squeeze(0)

        to_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])

        img1_grayscale = to_grayscale(img1)
        imgx_rgb = Image.fromarray(output_array).convert('RGB')
        imgx = loader(imgx1_rgb)
        edge = cannyEdgeDetector(imgx).detect()
        edge = edge[0]

        # find contours and generate polygon using skimage
        contours = find_contours(edge, 0, 'high')
        result_polygon = np.zeros(edge.shape + (3, ), np.uint8)
        for contour in contours:
            polygon = approximate_polygon(contour, tolerance=1)
            polygon = polygon.astype(np.int).tolist()
            for idx, coords in enumerate(polygon[:-1]):
                y1, x1, y2, x2 = coords + polygon[idx + 1]
                result_polygon = cv2.line(result_polygon, (x1, y1), (x2, y2), (255, 49, 49), 1)

        img1_np = np.array(img1)
        # Save the output masks to be poresented to the screen
        result_polygon = Image.fromarray(result_polygon)
        result_polygon.save('/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/edge.png','png')
        imgx1_rgb.save('/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/mask.png','png')

        colours = [[235, 27, 23],
              [27, 181, 13],
              [7, 219, 247]]
        # Make an estimate of the height of tree and present it to the screen
        area_array =[]
        image_array = []
        for i in range(0,len(prediction[0]['masks'])):
          pic = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
          ret, pic1 = cv2.threshold(pic, 200, 255, cv2.THRESH_BINARY)
          occurrences = pic1 == 255
          total = occurrences.sum()
          area_array.append(total)
          output_pic = Image.fromarray(pic1)
          output_pic_p = output_pic.convert("RGB")
          image_array.append(output_pic_p)

        num_pics = len(image_array)
        width = 600 
        height = 600

        # Colour the largest trees in so that they can be mactched up with the key.
        height_array = []
        for k in range (3):
          img = image_array[k]
          for i in range(0,width):# process all pixels
              for j in range(0,height):
                  data = img.getpixel((i,j))
                  if (data[0]==255 and data[1]==255 and data[2]==255):
                      img.putpixel((i,j),(colours[k][0],colours[k][1],colours[k][2]))


        pixel_area_to_area_in_meters =  0.00778102924
        zero = np.array(image_array[0])
        gradient_of_straight_line = 0.0573
        y_intersept = 7.5689

        max_gradient_of_straight_line = 0.0294
        max_y_intercept = 19

        min_gradient_of_straight_line = 0.0579
        min_y_intercept = 2
        
        area_array.sort(reverse=True)
        avarage_height_array =[]
        max_height_array = []
        min_height_array = []
        tree_area_array = []
        print(area_array)


        for c in range(3):
          zero += np.array(image_array[c])
          tree_area = area_array[c]*pixel_area_to_area_in_meters
          height = (tree_area *gradient_of_straight_line)+ y_intersept
          max_height = (tree_area*max_gradient_of_straight_line)+max_y_intercept
          min_height = (tree_area*min_gradient_of_straight_line)+min_y_intercept
          avarage_height_array.append(round(height,2))
          max_height_array.append(round(max_height,2))
          min_height_array.append(round(min_height,2))
          tree_area_array.append(tree_area)
        zero_image = Image.fromarray(zero)
        zero_image.save('/Users/Hadley 1/Documents/MEng Project/python/local/mapsamples/height.png','png')
        # Return all the predictions to be presented on the screen
        height_array.append(avarage_height_array)
        height_array.append(max_height_array)
        height_array.append(min_height_array)
        height_array.append(number_of_trees_in_image)
        height_array.append(tree_area_array) 
        
        return height_array


def get_model_instance_segementation(num_classes):
        # load an instance model pretrained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # get number of input features from classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pretrained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # get number of input features from the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model

   
        
        # compute image gradient
def gradient_img(img):
        img = img.squeeze(0)
        ten = torch.unbind(img)
        x = ten[0].unsqueeze(0).unsqueeze(0)

        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        G_x = conv1(Variable(x)).data.view(1, x.shape[2], x.shape[3])

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        G_y = conv2(Variable(x)).data.view(1, x.shape[2], x.shape[3])

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        # theta = np.arctan2(G_y, G_x)
        return G
    
    
# Driver fucntion
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
