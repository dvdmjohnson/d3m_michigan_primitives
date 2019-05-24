'''
GOTURN (Generic Object Tracking Using Regression Networks) is siamese network that uses the first five convolutional layers of CaffeNet. 
The two inputs of the network are the previous frame and the current frame, they both share convolutional layers. The features are then concatenated and passed through three fully connected layers
The output from the last fc layer are four numbers describing the bounding box position (x1, y1, x2, y2). Top left pixel coordinates and bottom right pixel coordinates.

Held, David and Thrun, Sebastian and Savarese, Silvio. Learning to track at 100 fps with deep regression networks. ECCV, 2016.
'''

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models

#Initialize weights of fully connected layers
def init_weights(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(1.0) #Initialize biases with 1
        m.weight.data.normal_(0, 0.005) #Initialize weights with normal distribution, zero mean and standard deviation of 0.005

#LRN code borrowed from: https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, k=True):
        super(LRN, self).__init__()
        self.k = k
        if self.k:
            self.average = nn.AvgPool3d(kernel_size=(local_size,1,1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.k:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class GoTurn(nn.Module):

    def __init__(self, train=True, use_gpu=True):
        super().__init__()

        #Generate features for both images individually
        self.features = nn.Sequential(
                nn.Conv2d(3,96,kernel_size=11,stride=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75, k=True),
                nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2,dilation=1,groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75, k=True),
                nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1,dilation=1,groups=2),
                nn.ReLU(),
                nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1,dilation=1,groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True))

        self.classifier = nn.Sequential(
                nn.Linear(256*6*6*2, 4096), #Two image inputs
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, 4))

        if train:
            #Freeze conv layers during training, initialized with ImageNet weights
            for param in self.features.parameters():
                param.requires_grad = False

            #Apply initialization weights
            self.classifier.apply(init_weights)

    def forward(self,x0,x1):
        x0 = self.features(x0)
        x0 = x0.view(-1,256*6*6)
        x1 = self.features(x1)
        x1 = x1.view(-1,256*6*6)

        x = torch.cat((x0,x1),1)
        out = self.classifier(x)
        #output is in the format [x1,y1,x2,y2]
        return out*10 #scale output by 10 per paper

#Adopted from: https://github.com/davheld/GOTURN/blob/5d4fa8193792f493d2a443f472874692fa430533/src/helper/bounding_box.cpp
kContextFactor = 2.0 
kScaleFactor = 10.0 #Scale output from network by this amount

#If true, bounding boxes are [xtl, ytl, xbr, ybr]
#If false, bounding boxes are [center_x, center_y, width, height]
use_coordinate_outputs = True

class BoundingBox:

    def __init__(self, bounding_box=None):

        self.scale_factor = kScaleFactor

        if not bounding_box is None:
            if not len(bounding_box) is 4:
                raise Exception('Boundingbox vector has {} elements'.format(len(bounding_box)))
            
            if use_coordinate_outputs:
                self.x1 = bounding_box[0]
                self.y1 = bounding_box[1]
                self.x2 = bounding_box[2]
                self.y2 = bounding_box[3]
            else:
                center_x = bounding_box[0]
                center_y = bounding_box[1]
                width = bounding_box[2]
                height = bounding_box[3]

                self.x1 = center_x - width/2
                self.y1 = center_y - height/2
                self.x2 = center_x + width/2
                self.y2 = center_y + height/2

    def getVector(self):
        bounding_box = []

        if use_coordinate_outputs:
            bounding_box.append(self.x1)
            bounding_box.append(self.y1)
            bounding_box.append(self.x2)
            bounding_box.append(self.y2)
        else:
            bounding_box.append(self.getCenterX())
            bounding_box.append(self.getCenterY())
            bounding_box.append(self.getWidth())
            bounding_box.append(self.getHeight())

        return bounding_box
                
    def getRegion(self):
        region = collections.namedtyple('region', ['x', 'y', 'width', 'height'])

        region.x = self.x1
        region.y = self.y2
        region.width = self.getWidth()
        region.height = self.getHeight()

    def scale(self, image, scale=None):

        if scale is None:
            scale = self.scale_factor

        width = image.width
        height = image.height

        #Normalize coordinates to be between 0 and 1
        self.x1 /= width
        self.y1 /= height
        self.x2 /= width
        self.y2 /= height

        #Scale from 0 to scale factor
        self.x1 *= scale
        self.x2 *= scale 
        self.y1 *= scale 
        self.y2 *= scale 

    def unscale(self, image):
        width = image.width
        height = image.height

        #Normalize coordinates to be between 0 and 1
        self.x1 /= self.scale_factor
        self.x2 /= self.scale_factor
        self.y1 /= self.scale_factor
        self.y2 /= self.scale_factor

        self.x1 *= width
        self.y1 *= height
        self.x2 *= width
        self.y2 *= height

    #Location of bounding box relative to original image
    def recenter(self, search_location, edge_spacing_x, edge_spacing_y):
        self.x1 = self.x1 - search_location.x1 + edge_spacing_x
        self.y1 = self.y1 - search_location.y1 + edge_spacing_y
        self.x2 = self.x2 - search_location.x1 + edge_spacing_x
        self.y2 = self.y2 - search_location.y1 + edge_spacing_y

    def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
        self.x1 = max(0, self.x1 + search_location.x1 - edge_spacing_x)
        self.y1 = max(0, self.y1 + search_location.y1 - edge_spacing_y)
        self.x2 = min(raw_image.width, self.x2 + search_location.x1 - edge_spacing_x)
        self.y2 = min(raw_image.height, self.y2 + search_location.y1 - edge_spacing_y)

    def compute_output_width(self):
        bbox_width = self.x2 - self.x1

        output_width = kContextFactor * bbox_width

        return max(1.0, output_width)

    def compute_output_height(self):
        bbox_height = self.y2 - self.y1

        output_height = kContextFactor * bbox_height

        return max(1.0, output_height)

    def getWidth(self):
        return self.x2 - self.x1

    def getHeight(self):
        return self.y2 - self.y1

    def getCenterX(self):
        return (self.x1 + self.x2) / 2

    def getCenterY(self):
        return (self.y1 + self.y2) / 2

    def get_edge_spacing_x(self):
        output_width = self.compute_output_width()
        bbox_center_x = self.getCenterX()

        return max(0, output_width/2 - bbox_center_x)

    def get_edge_spacing_y(self):
        output_height = self.compute_output_height()
        bbox_center_y = self.getCenterY()

        return max(0, output_height/2 - bbox_center_y)

    def __repr__(self):
        bounding_box = self.getVector()

        return "(%s, %s, %s, %s)" % (str(bounding_box[0]), str(bounding_box[1]),str(bounding_box[2]), str(bounding_box[3]))

#Portions adopted from: https://github.com/davheld/GOTURN/blob/master/src/helper/image_proc.cpp

#Crop PIL image and return Tensor image
def crop_image(pil_image):
    loader = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])

    tensor_image = loader(pil_image)
    if tensor_image.shape[0] == 1: #If image is grayscale, then repeat that color channel
        tensor_image = tensor_image.repeat(3,1,1)

    return tensor_image #return tensor image


def ComputeCropPadImageLocation(bbox_tight, image):
    bbox_center_x = bbox_tight.getCenterX()
    bbox_center_y = bbox_tight.getCenterY()

    image_width = image.width
    image_height = image.height

    output_width = bbox_tight.compute_output_width()
    output_height = bbox_tight.compute_output_height()

    roi_left = max(0, bbox_center_x - output_width/2)
    #TODO: Call this roi_top instead of roi_bottom, but first figure out why it's called bottom
    #Possibly the pixels start counting from the bottom left corner instead of the top left corner
    roi_bottom = max(0, bbox_center_y - output_height/2)

    left_half = min(output_width/2, bbox_center_x)
    right_half = min(output_width/2, image_width - bbox_center_x)

    roi_width = max(1.0, left_half+right_half)

    top_half = min(output_height/2, bbox_center_x)
    bottom_half = min(output_height/2, image_height - bbox_center_y)

    roi_height = max(1.0, top_half+bottom_half)

    pad_image_location = BoundingBox()
    pad_image_location.x1 = roi_left
    pad_image_location.y1 = roi_bottom
    pad_image_location.x2 = roi_left + roi_width
    pad_image_location.y2 = roi_bottom + roi_height

    return pad_image_location

def CropPadImage(bbox_tight, image, edge_spacing_x=None, edge_spacing_y=None):
    pad_image_location = ComputeCropPadImageLocation(bbox_tight, image) 

    image_width = image.width
    image_height = image.height

    roi_left = min(pad_image_location.x1, image_width-1)
    roi_bottom = min(pad_image_location.y1, image_height-1)
    roi_width = min(image_width, max(1.0, np.ceil(pad_image_location.x2 - pad_image_location.x1)))
    roi_height = min(image_height, max(1.0, np.ceil(pad_image_location.y2 - pad_image_location.y1)))

    #Perform crop
    #To note: round(a + b) != round(a) + round(b). Example: a=117.5 b=175
    myRoi = (round(roi_left), round(roi_bottom), round(roi_left)+round(roi_width), round(roi_bottom)+round(roi_height))
    cropped_image = image.crop(myRoi)

    output_width = max(np.ceil(bbox_tight.compute_output_width()), roi_width)
    output_height = max(np.ceil(bbox_tight.compute_output_height()), roi_height)
    output_image = Image.new(image.mode,(int(output_width), int(output_height))) 

    edge_spacing_x = min(bbox_tight.get_edge_spacing_x(), image_width-1)
    edge_spacing_y = min(bbox_tight.get_edge_spacing_y(), image_height-1)

    output_rect = (int(round(edge_spacing_x)), int(round(edge_spacing_y)), int(round(edge_spacing_x)+round(roi_width)), int(round(edge_spacing_y)+round(roi_height)))
    try:
        output_image.paste(cropped_image, output_rect)
    except:
        print('ValueError: tools/utils.py: cropped_image: {}, output_rect: {}'.format(cropped_image.size, output_rect))

    pad_image = output_image 
    return pad_image, pad_image_location, edge_spacing_x, edge_spacing_y;

# Update key names to match the pretrained model 
def convert_state_dict_keys(tracker_state_dict, goturn=False):
    new_state_dict = {}

    feature_layers = 10 #First 10 keys belong to the conv layers
    param_count = 0 

    for key in tracker_state_dict.keys():
        if param_count < feature_layers:
            new_state_dict['features.'+key] = tracker_state_dict[key]
        param_count += 1

    return new_state_dict

