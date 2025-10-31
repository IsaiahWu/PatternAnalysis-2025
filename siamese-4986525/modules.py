"""
model.py

This module defines the Siamese neural network architecture and the contrastive loss function
for skin lesion similarity measurement or classification based on image embeddings.


Author: Wu Chun Yueh
Date: 31st October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):

  """
    Siamese Neural Network for learning image embeddings and comparing image similarity.
    
    Here uses a shared convolutional base to process paired images independently, 
    Output embedding vectors for each image.
    
    Architecture:
        - 3 Convolutional blocks (Conv2d -> ReLU -> MaxPool)
        - Fully connected layers for embedding generation
        - Dropout reduce overfitting
    Args:
        embedding_dim (int): Dimension of the output embedding vector. Default: 128
  """

    
  def __init__(self, embedding_dim=128):
    super(SiameseNetwork, self).__init__()

    self.conv1 = nn.Conv2d (in_channels = 3, out_channels = 16, kernel_size = 3) # Defining my fliter
    self.conv2 = nn.Conv2d (in_channels = 16, out_channels = 32, kernel_size = 3)
    self.conv3 = nn.Conv2d (in_channels = 32, out_channels = 64, kernel_size = 3)

    self.pool = nn.MaxPool2d (kernel_size = 2, stride = 2) #Maxpooling, retriving the largest input value
    self.relu = nn.ReLU() #Make negative to 0


    self.flatten = nn.Flatten() #Make it to 1 dimension

    self.fc1 = nn.Linear(12544, 512) # First layer of neurons learn, general compressed features, 64*16*16
    self.fc_embedding = nn.Linear(512, embedding_dim) # Second layer of nerons
    self.dropout = nn.Dropout(0.5) #help reduce overfitting


  def forward_once(self, x):

    """
        Define the process flow of a single image through the network to extract its embedding.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Embedding vector of shape (batch_size, embedding_dim)
    """  
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)


    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.pool(x)

    # Flatten
    x = self.flatten(x)


    # Input values to neurons
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc_embedding(x)

    return x


  def forward(self, input1, input2):

    """
      Process a pair of images through the network and output emedding vecctors 

      Args:
            input1 (torch.Tensor): First image tensor of shape (batch_size, 3, height, width)
            input2 (torch.Tensor): Second image tensor of shape (batch_size, 3, height, width)
            
      Returns:
            tuple: (embedding1, embedding2)
    """

    embedding1 = self.forward_once(input1)
    embedding2 = self.forward_once(input2)
    return embedding1, embedding2


# Contrastive Loss
class ContrastiveLoss(nn.Module):

  """
    Calculate the Contrastive Loss function for training Siamese Networks.

    Goal:
      - Minimize distance between embeddings of similar images 
      - Maximize distance between embeddings of dissimilar images 
    
    The loss uses Euclidean distance in the embedding space with a margin input
    
    Args:
        margin (float): Margin threshold for negative pairs. Distances beyond this margin will not contribute to the loss
                       
    Formula:
        L = (1 - Y) * D^2 + Y * max(margin - D, 0)^2
        where D is Euclidean distance, Y is label (0=similar, 1=dissimilar)
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss
