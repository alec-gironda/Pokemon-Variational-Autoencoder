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
plt.savefig("./original.jpg")

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_layer_1 =  nn.Conv2d(1,32,3)
        self.max_pool_1 = nn.MaxPool2d(3)

        self.conv_layer_2 =  nn.Conv2d(32,64,3)
        self.max_pool_2= nn.MaxPool2d(3)

        self.conv_layer_3 =  nn.Conv2d(64,128,3)
        self.max_pool_3= nn.MaxPool2d(3)

        self.conv_layer_4 =  nn.Conv2d(128,256,3)
        self.max_pool_4= nn.MaxPool2d(3)

        self.flatten = nn.Flatten()

        self.fc_layer_1 = nn.Linear(8192,128)

    def forward(self, x):

        x = self.conv_layer_1(x)
        x = self.max_pool_1(x)

        x = self.conv_layer_2(x)
        x = self.max_pool_2(x)

        x = self.conv_layer_3(x)
        x = self.max_pool_3(x)

        x = self.flatten(x)

        x = self.fc_layer_1(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc_layer_1 = nn.Linear(128,16777216)


        self.conv_layer_1 =  nn.Conv2d(256,128,3)
        self.max_pool_1 = nn.MaxPool2d(3)

        self.conv_layer_2 =  nn.Conv2d(128,64,3)
        self.max_pool_2 = nn.MaxPool2d(3)

        self.fc_layer_2 = nn.Linear(46656,65536)




    def forward(self, x):

        x = self.fc_layer_1(x)

        x = torch.reshape(x,(819,256,256,256))

        x = self.conv_layer_1(x)
        x = self.max_pool_1(x)

        x = self.conv_layer_2(x)
        x = self.max_pool_2(x)

        x = torch.reshape(x,(819,64*27*27))

        x = self.fc_layer_2(x)

        x = torch.reshape(x,(819,1,256,256))

        return x


def loss_fn(input_image,output_image):
    input_image = input_image.view(-1, 256*256)
    output_image = output_image.view(-1, 256*256)
    return torch.sum((input_image-output_image)**2)

ims = torch.reshape(ims,(819,1,256,256))

encoder = Encoder()
decoder = Decoder()

loss =  loss_fn # Step 2: loss
encoder_opt = torch.optim.Adam(encoder.parameters(), lr=.001) # Step 3: training method
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=.001) # Step 3: training method


train_loss_history = []
for epoch in range(100):
    train_loss = 0.0
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    encoded_out = encoder(ims)
    decoded_out = decoder(encoded_out)
    fit = loss(ims,decoded_out)
    fit.backward()
    encoder_opt.step()
    decoder_opt.step()
    train_loss += fit.item()
    train_loss_history.append(train_loss)
    print(f'Epoch {epoch}, Train loss {train_loss}')
print(train_loss_history[-1])


test = torch.reshape(ims[0],(1,1,256,256))
out = model(test)
out = torch.reshape(out,(256,256))

plt.imshow(transforms.ToPILImage()(out))
plt.savefig("./out.jpg")
