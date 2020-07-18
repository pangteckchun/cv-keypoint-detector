## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        # This will result in output tensor of (32, 222, 222); output_dim = (W-F)/S + 1 = (224-3)/1 + 1 = 222
        # After one max pool becomes (32, 111, 111)
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        # 2nd convo layer, 32 inputs, 64 outputs channels/feature maps, 3x3 square convolution kernel
        # This will result in output tensor of (64, 109, 109); output_dim = (W-F)/S + 1 = (111-3)/1 + 1 = 109
        # After one more max pool becomes (68, 54, 54), where 109/2 = 54.5 round down to 54
        self.conv2 = nn.Conv2d(32, 64, 3)

         # 3rd convo layer, 64 inputs, 128 outputs channels/feature maps, 3x3 square convolution kernel
        # This will result in output tensor of (128, 54, 54); output_dim = (W-F)/S + 1 = (54-3)/1 + 1 = 52
        # After one more max pool becomes (68, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # Max pool layer with kernel size of 2 and stride 2
        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layer, 128 inputs from last convo layer with 26*26 filter size (post max pool) and 136 outputs (2 for each of the 68 facial keypoints)
        self.fcl1 = nn.Linear(128*26*26, 136)

        # 2nd fully connected layer
        self.fcl2 = nn.Linear(136, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two convo + max pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # FCL layer
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
