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
import torch.nn as nn


num_simpsons = len(os.listdir("./simpsons")[:5000])

print(num_simpsons)

ims = []

for im_name in os.listdir("./simpsons")[:5000]:
    s = (f"./simpsons/{str(im_name)}")
    curr = torchvision.io.read_image(s)
    ims.append(curr)

ims = torch.stack(ims)

ims = ims/255

print(ims.shape)

plt.imshow(transforms.ToPILImage()(ims[8]))
plt.savefig("./original.png")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()

        self.conv_layer_1 =  nn.Conv2d(3,16,5,stride = 2,padding = 2)
        self.bn_1 = nn.BatchNorm2d(16)

        self.conv_layer_2 =  nn.Conv2d(16,32,5,stride = 2,padding = 2)
        self.bn_2 = nn.BatchNorm2d(32)

        self.conv_layer_3 =  nn.Conv2d(32,64,5,stride = 2,padding = 2)
        self.bn_3 = nn.BatchNorm2d(64)

        self.conv_layer_4 =  nn.Conv2d(64,128,5,stride = 2,padding = 2)
        self.bn_4 = nn.BatchNorm2d(128)

        self.conv_layer_5 =  nn.Conv2d(128,256,5,stride = 2,padding = 2)
        self.bn_5 = nn.BatchNorm2d(256)

        self.conv_layer_6 =  nn.Conv2d(256,512,5,stride = 2,padding = 2)
        self.bn_6 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()

        self.mean = nn.Linear(8192,128)
        self.log_var = nn.Linear(8192,128)

    def forward(self, x):

        x = self.conv_layer_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_layer_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_layer_3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_layer_4(x)
        x = self.bn_4(x)
        x = self.relu(x)

        x = self.conv_layer_5(x)
        x = self.bn_5(x)
        x = self.relu(x)

        x = self.conv_layer_6(x)
        x = self.bn_6(x)
        x = self.relu(x)

        x = self.flatten(x)

        mu = self.mean(x)
        var = self.log_var(x)


        return mu,var

class Decoder(nn.Module):
    def __init__(self,batch_size):
        super(Decoder, self).__init__()

        self.relu = nn.ReLU()

        self.fc_layer_1 = nn.Linear(128,8192)

        self.conv_layer_1 =  nn.ConvTranspose2d(512,256,4,stride = 2,padding =1)
        self.bn_1 = nn.BatchNorm2d(256)

        self.conv_layer_2 =  nn.ConvTranspose2d(256,128,4,stride = 2,padding =1)
        self.bn_2 = nn.BatchNorm2d(128)

        self.conv_layer_3 =  nn.ConvTranspose2d(128,64,4,stride = 2,padding =1)
        self.bn_3 = nn.BatchNorm2d(64)

        self.conv_layer_4 =  nn.ConvTranspose2d(64,32,4,stride = 2,padding =1)
        self.bn_4 = nn.BatchNorm2d(32)

        self.conv_layer_5 =  nn.ConvTranspose2d(32,16,4,stride = 2,padding =1)
        self.bn_5 = nn.BatchNorm2d(16)

        self.conv_layer_6 =  nn.ConvTranspose2d(16,3,4,stride = 2,padding =1)

        self.flatten = nn.Flatten()

        self.batch_size = batch_size


    def forward(self, x):

        x = self.fc_layer_1(x)

        x = self.relu(x)

        x = torch.reshape(x,(self.batch_size,512,4,4))

        x = self.conv_layer_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_layer_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        x = self.conv_layer_3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.conv_layer_4(x)
        x = self.bn_4(x)
        x = self.relu(x)

        x = self.conv_layer_5(x)
        x = self.bn_5(x)
        x = self.relu(x)

        x = self.conv_layer_6(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = torch.clamp(x,min=0,max=1)

        x = torch.reshape(x,(self.batch_size,3,256,256))

        return x

from torch.autograd import Variable

def middlePart(mu,var):
  sd = torch.exp(var * 0.5)
  e = Variable(torch.randn(sd.size()).cuda())
  z = e.mul(sd).add_(mu)

  return z


def criterion(x_out, target, z_mean, z_logvar, alpha=1, beta=1):
    """
    Criterion for VAE done analytically
    output: loss
    output: bce
    output: KL Divergence
    """
    bce = F.mse_loss(x_out, target, size_average=False) #Use MSE loss for images
    kl = -0.5 * torch.sum(1 + z_logvar - (z_mean**2) - torch.exp(z_logvar)) #Analytical KL Divergence - Assumes p(z) is Gaussian Distribution
    loss = ((alpha * bce) + (beta * kl)) / x_out.size(0)
    return loss

batch_size = 10
new_ims = ims

trainDataLoader = torch.utils.data.DataLoader(new_ims,batch_size=batch_size,shuffle=True)

encoder = Encoder().to("cuda")
decoder = Decoder(batch_size).to("cuda")

loss =  criterion # Step 2: loss

# https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs

import torch.optim

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=.001) # Step 3: training method
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=.001) # Step 3: training method

# e_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, step_size=100, gamma=0.1)
# d_scheduler = torch.optim.lr_scheduler.StepLR(decoder_opt, step_size=100, gamma=0.1)

train_loss_history = []
for epoch in range(2000):
  for i, curr_ims in enumerate(trainDataLoader):
    train_loss = 0.0
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    mu, var = encoder(curr_ims.to("cuda"))
    x = middlePart(mu,var)
    decoded_out = decoder(x)
    fit = loss(curr_ims.to("cuda"),decoded_out,mu,var)
    fit.backward()
    encoder_opt.step()
    decoder_opt.step()
    # e_scheduler.step()
    # d_scheduler.step()

    train_loss += fit.item()
    train_loss_history.append(train_loss)
    if epoch % 10 == 0:
      print(f'Epoch {epoch}, Train loss {train_loss}')
print(train_loss_history[-1])

test = new_ims[:batch_size].to("cuda")
mu,var = encoder(test)
x = middlePart(mu,var)
out = decoder(x)
out = torch.reshape(out,(batch_size,3,256,256))

plt.imshow(transforms.ToPILImage()(out[8]))
plt.savefig("./out.png")

torch.save(decoder.state_dict(), "./decoder_mod")

decoder = Decoder(batch_size).to("cuda")
decoder.load_state_dict(torch.load("./decoder_mod"))

out = decoder(x)
out = torch.reshape(out,(batch_size,3,256,256))

plt.imshow(transforms.ToPILImage()(out[8]))
plt.savefig("./out_after_load.png")
