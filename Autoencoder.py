#imports

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np
import os

num_pokemon = len(os.listdir("data/pokemon_jpg/pokemon_jpg/"))

ims = []

for im_name in os.listdir("data/pokemon_jpg/pokemon_jpg/"):
    s = (f"data/pokemon_jpg/pokemon_jpg/{str(im_name)}")
    curr = torchvision.io.read_image(s)
    ims.append(curr)

ims = torch.stack(ims)

ims = ims[:,1,:,:]

ims = ims/255

plt.imshow(transforms.ToPILImage()(ims[0]))
plt.savefig("/Users/alecgironda/Desktop/W23/CSCI1051/Pokemon-Variational-Autoencoder/original.jpg")

import torch.nn as nn

#nn architecture

class FC_NN(nn.Module):
    def __init__(self):
        super(FC_NN, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(256*256,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128,256),
        nn.ReLU(),
        nn.Linear(256,256*256)
        )

    def forward(self, x):
        x = x.view(-1, 256*256)
        return self.model(x)

#nn architecture

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer_1 =  nn.Conv2d(1,3,3)
        self.max_pool_1 = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()

        self.fc_layer_1 = nn.Linear(7056,128)
        self.relu = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128,7056)

        self.conv_layer_2 = nn.Conv2d(3,1,3)

        self.fc_layer_3 = nn.Linear(82*82,256*256)

    def forward(self, x):

        x = self.conv_layer_1(x)
        x = self.max_pool_1(x)
        x = self.flatten(x)

        x = self.fc_layer_1(x)
        x = self.relu(x)
        x = self.fc_layer_2(x)

        x = torch.reshape(x,(3,84,84))

        x = self.conv_layer_2(x)

        x = self.flatten(x)

        x = self.fc_layer_3(x)

        return x

def loss_fn(input_image,output_image):
    input_image = input_image.view(-1, 256*256)
    return torch.sum((input_image-output_image)**2)

ims = torch.reshape(ims,(819,1,256,256))

model = CNN()
loss =  loss_fn # Step 2: loss
optimizer = torch.optim.Adam(model.parameters(), lr=.001) # Step 3: training method

train_loss_history = []
for epoch in range(1):
    train_loss = 0.0
    optimizer.zero_grad()
    predicted_output = model(ims[0])
    fit = loss(ims[0],predicted_output)
    fit.backward()
    optimizer.step()
    train_loss += fit.item()
    train_loss_history.append(train_loss)
    print(f'Epoch {epoch}, Train loss {train_loss}')
print(train_loss_history[-1])


out = model(ims[0])
out = torch.reshape(out,(256,256))

plt.imshow(transforms.ToPILImage()(out))
plt.savefig("/Users/alecgironda/Desktop/W23/CSCI1051/Pokemon-Variational-Autoencoder/out.jpg")
