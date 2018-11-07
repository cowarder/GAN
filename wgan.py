import torchvision.datasets as datasets
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import os
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=200,
                   help='epoch of training (default:200)')
parser.add_argument('--lr', type=float, default=0.00005,
                   help='learning rate (default:0.00005)')
parser.add_argument('--latent_dim', type=int, default=100,
                    help='length of random vector (default:100)')
parser.add_argument('--channels', type=int, default=1,
                   help='channels of image (default:1)')
parser.add_argument('--img_size', type=int, default=28,
                   help='image size (default:28)')
parser.add_argument('--n_critic', type=int, default=5,
                   help='how many times of training D before training G (default:5)')
parser.add_argument('--batch_size', type=int, default=64,
                   help='batch size of training (default:64)')
parser.add_argument('--clip', type=float, default=0.01,
                   help='clipping parameter (default:0.01)')
parser.add_argument('--log_interval', type=int, default=1000,
                   help='how many batches before logging training data (default:100)')
				   
args = parser.parse_args()
#print(args)

img_size = (args.channels, args.img_size, args.img_size)


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(args.latent_dim, 128)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 256)
        self.norm2 = nn.BatchNorm1d(256, 0.8)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(256, 512)
        self.norm3 = nn.BatchNorm1d(512, 0.8)
        self.relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(512, 1024)
        self.norm4 = nn.BatchNorm1d(1024, 0.8)
        self.relu4 = nn.LeakyReLU(0.2)
        self.fc5 = nn.Linear(1024, np.prod(img_size))
        self.output = nn.Tanh()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.norm2(self.fc2(x)))
        x = self.relu3(self.norm3(self.fc3(x)))
        x = self.relu4(self.norm4(self.fc4(x)))
        x = self.fc5(x)
        x = self.output(x)
        x = x.view(x.size(0), *(img_size))
        return x
		
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(np.prod(img_size), 512)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        score = self.output(x)
        return score
        
		
dataloader = DataLoader(
    datasets.MNIST('./mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])),
    batch_size = args.batch_size, shuffle=True)

generator = Generator()
discriminator = Discriminator()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr = args.lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr = args.lr)

batches_done = 0
for epoch in range(1,args.n_epoch+1):
	for batch_idx, (imgs, _) in enumerate(dataloader):
		#train discriminator
		imgs = imgs.to(device)
		z = torch.randn(imgs.size(0), args.latent_dim).to(device)
		gen_imgs = generator(z).detach()
		optimizer_D.zero_grad()
		
		loss_D = -torch.mean(discriminator(imgs)) + torch.mean(discriminator(gen_imgs))
		loss_D.backward()
		optimizer_D.step()
		
		for param in discriminator.parameters():
			param.data.clamp_(-args.clip, args.clip)
		
		#train generator
		if batch_idx % args.n_critic == 0:
			z = torch.randn(imgs.size(0), args.latent_dim).to(device)
			gen_imgs = generator(z)
			optimizer_G.zero_grad()
			
			loss_G = -torch.mean(discriminator(gen_imgs))
			loss_G.backward()
			optimizer_G.step()
		if batches_done % args.log_interval == 0:
			save_image(gen_imgs.data[:25], './mnist/generateddata/%d.png' % batches_done, nrow=5, normalize=True)
		batches_done += 1
	print("Loss of Generator:{}, loss of Discriminator:{}".format(loss_G, loss_D))