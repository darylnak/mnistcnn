"""
contains all network structure definitions
including layers definition and forward pass function definitions
"""
# PyTorch and neural network imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import math

#-------------------------------------------------#
#-------------------------------------------------#
#---------------Baseline Variables----------------#
#-------------------------------------------------#
#-------------------------------------------------#
# set the randomness to keep reproducible results
torch.manual_seed(0)
np.random.seed(0)

# 10 types of digits for classification
num_classes = 10

# MLP vars
mlp_input_size = 784            # input size to mlp network
mlp_output_size = num_classes
mlp_hidden_size = 35

# CNN vars
cnn_output_size = num_classes
cnn_input_size = 28
cnn_in_channels = 1
conv_kernel_size = 5 # filter size
conv_out_channels = 10
conv_stride = 1
conv_padding = 2
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#

def conv_out_dim_calc(in_size, out_channels, kernel_size, stride, padding):
    out_width = math.floor((in_size + (2 * padding) - kernel_size)/stride) + 1
    return (out_width**2) * out_channels

class BaselineMLP(nn.Module):
    def __init__(self):
        """
        A multilayer perceptron model
        Consists of one hidden layer and 1 output layer (all fully connected)
        """
        super(BaselineMLP, self).__init__()
        # a fully connected layer from input layer to hidden layer
        # mlp_input_size denotes number of input neurons
        # mlp_hiddent_size denotes number of hidden neurons
        self.fc1 = nn.Linear(mlp_input_size, mlp_hidden_size)

        # a fully connected layer from hidden layer to output layer
        # mlp_output_size denotes number of output neurons
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_output_size)

    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying
        logistic activation function after hidden layer.
        """
        # pass X from input layer to hidden layer
        out = self.fc1(X)

        # apply an activation function to the output of hidden layer
        out = torch.sigmoid(out)

        # pass output from hidden layer to output layer (Daryl: num classes)
        out = self.fc2(out)

        # return the feed forward output
        # don't need to apply another activation function here if
        # the loss function implements it
        return out
class BaselineCNN(nn.Module):
    def __init__(self):
        """
        A basic convolutional neural network model for baseline comparison.
        Consists of one Conv2d layer, followed by 1 fully-connected (FC) layer:
        conv1 -> fc1 (outputs)
        """
        super(BaselineCNN, self).__init__()
        self.conv_out_dim = conv_out_dim_calc(cnn_input_size, 
                                            conv_out_channels, 
                                            conv_kernel_size, 
                                            conv_stride, 
                                            conv_padding)

        self.n_classes = num_classes
        self.conv = nn.Conv2d(cnn_in_channels, 
                              conv_out_channels, 
                              kernel_size=conv_kernel_size, 
                              stride=conv_stride, 
                              padding=conv_padding)

        self.l1 = nn.Linear(self.conv_out_dim, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 300)
        self.l4 = nn.Linear(300, self.n_classes)
        self.tanh=nn.Tanh()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.
        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.
        You may need the function "num_fc_features" below to help implement
        this function
        Parameters: X --- an input batch of images
        Returns: out --- the output of the network
        """
        # define the forward function
        out = self.tanh(self.conv(X).view(-1, self.conv_out_dim))

        out = self.tanh(self.l1(out))
        out = self.dropout(out)

        out = self.tanh(self.l2(out))
        out = self.dropout(out)

        out = self.tanh(self.l3(out))
        out = self.dropout(out)

        out = self.l4(out)

        return out

    """
    Count the number of flattened features to be passed to fully connected layers
    Parameters: inputs --- 4-dimensional [batch x num_channels x conv width x conv height] 
                           output from the last conv layer
    Return: num_features --- total number of flattened features for the last layer
    """
    def num_fc_features(self, inputs):
        # Get the dimensions of the layers excluding the batch number
        size = inputs.size()[1:]

        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

#-------------------------------------------------#
#-------------------------------------------------#
#-------------Modified CNN Variables--------------#
#-------------------------------------------------#
#-------------------------------------------------#
kernelDim = 3
numFilters = 40
strideLength = 1
paddingSize = 1
dataInChannels = 1
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#

class CNNPlusPlus(nn.Module):

    def calcuPoolOutDim(in_size, out_channels, kernel_size, stride):
        out_width = math.floor((in_size - kernel_size)/stride) + 1

        return (out_width**2) * out_channels
    
    def __init__(self):

        super(CNNPlusPlus, self).__init__()

        # Conv 1 Pooling 1
        self.conv1 = nn.Conv2d(dataInChannels,
                            numFilters,
                            kernel_size = kernelDim,
                            stride = strideLength,
                            padding = paddingSize)
        self.pool1 = nn.MaxPool2d(2, stride=1)

        # Conv 2 with Pooling 2
        self.conv2 = nn.Conv2d(numFilters,
                            2*numFilters,
                            kernel_size = kernelDim,
                            stride = strideLength,
                            padding = paddingSize)
        self.pool2 = nn.MaxPool2d(2, stride=1)

        # Conv 3 with Pooling 3
        self.conv3 = nn.Conv2d(2*numFilters,
                            4*numFilters,
                            kernel_size=conv_kernel_size,
                            stride=conv_stride,
                            padding=conv_padding)
        self.pool3 = nn.MaxPool2d(2, stride=1)

        # Conv 4 with Pooling 4
        self.conv4 = nn.Conv2d(4*numFilters,
                            8*numFilters,
                            kernel_size=conv_kernel_size,
                            stride=conv_stride,
                            padding=conv_padding)
        self.pool4 = nn.MaxPool2d(2, stride=1)

        self.l1 = nn.Linear(216320, 1000).cuda()
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 200)
        self.l4 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()
        self.dropoutIn = nn.Dropout2d(p=0.5)
        self.dropoutHid = nn.Dropout2d(p=0.5)

    def forward(self, X):
        """
        Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.
        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.
        
        Parameters: X --- an input batch of images
        Returns:    out --- the output of the network
        """
        # define the forward function

        # Conv 1
        out = self.conv1(X)
        out = self.relu(out)
    
        # Conv 2 and pool
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)

        # Conv 3
        out = self.conv3(out)
        out = self.relu(out)

        # Conv 4 and pool
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool4(out)

        # Flatten output to feed into network
        out = self.tanh(out.view(-1, self.calcuConvOutDim(out)))
        
        # Hidden layer 1
        out = self.l1(out)
        out = self.relu(out)
        out = self.dropoutHid(out)

        # Hidden layer 2
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropoutHid(out)

        # Hidden layer 3
        out = self.l3(out)
        out = self.relu(out)
        out = self.dropoutHid(out)

        # Output layer
        out = self.l4(out)

        return out

    """
    Count the number of flattened features to be passed to fully connected layers
    Parameters: inputs --- 4-dimensional [batch x num_channels x conv width x conv height]
                           output from the last conv layer
    Return: num_features --- total number of flattened features for the last layer
    """
    def calcuConvOutDim(self, inputs):
        # Get the dimensions of the layers excluding the batch number
        size = inputs.size()[1:]

        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features