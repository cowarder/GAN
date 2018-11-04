import argparse
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200,
	help="epochs of training (default:200)")
parser.add_argument('--lr', type=float, default=0.0002,
	help='adam learning rate (default:0.0002)')
parser.add_argument('--log_interval', type=int ,default=1000,
	help='how many batch sizes before logging info (default:20)')
parser.add_argument('--batch_size', type=int, default=64,
	help='batch size (default:64)')
parser.add_argument('--n_cpu', type=int, default=8,
	help='number of cpu threads during batch generation')
parser.add_argument('--channels', type=int, default=1,
	help='number of image channels')
parser.add_argument('--img_size', type=int, default=28,
	help='size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=100,
	help='dimensionality of the latent space')
parser.add_argument('--b1', type=float, default=0.5,
	help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
	help='adam: decay of first order momentum of gradient')
	
args = parser.parse_args()
img_shape = (args.channels, args.img_size, args.img_size)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
	
	def __init__(self):
		super(Generator, self).__init__()
		
		self.fc1 = nn.Linear(args.latent_dim, 128)
		self.relu1 = nn.LeakyReLU(0.2)
		self.fc2 = nn.Linear(128, 256)
		self.norm1 =  nn.BatchNorm1d(256,0.8)
		self.relu2 = nn.LeakyReLU(0.2)
		self.fc3 = nn.Linear(256, 512)
		self.norm2 =  nn.BatchNorm1d(512,0.8)
		self.relu3 = nn.LeakyReLU(0.2)
		self.fc4 = nn.Linear(512, 1024)
		self.norm3 =  nn.BatchNorm1d(1024,0.8)
		self.relu4 = nn.LeakyReLU(0.2)
		self.fc5 = nn.Linear(1024, int(np.prod(img_shape)))
		self.output = nn.Tanh()
	
	def forward(self, x):
		x = self.relu1(self.fc1(x))
		x = self.relu2(self.norm1(self.fc2(x)))
		x = self.relu3(self.norm2(self.fc3(x)))
		x = self.relu4(self.norm3(self.fc4(x)))
		x = self.output(self.fc5(x))
		x = x.view(x.size(0), *img_shape)
		return x
	
class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()
		self.fc1=nn.Linear(int(np.prod(img_shape)), 512)
		self.relu1 = nn.LeakyReLU(0.2)
		self.fc2 = nn.Linear(512, 256)
		self.relu2 = nn.LeakyReLU(0.2)
		self.fc3 = nn.Linear(256, 1)
		self.output = nn.Sigmoid()
		
	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.relu1(self.fc1(x))
		x = self.relu2(self.fc2(x))
		score = self.output(self.fc3(x))
		return score
		

def main():

	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	#构建生成器和判别器模型
	generator = Generator()
	discriminator = Discriminator()
	generator = generator.to(device)
	discriminator = discriminator.to(device)

	#加载数据
	os.makedirs('mnist', exist_ok=True)
	dataloader = DataLoader(
		datasets.MNIST('mnist', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
			])),
		batch_size=args.batch_size, shuffle=True)

	#定义损失函数
	adversarial_loss = nn.BCELoss()
	adversarial_loss = adversarial_loss.to(device)
		
	#定义优化器
	optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
	optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

	#开始训练
	for epoch in range(args.epochs):

		print("Epoch:{}".format(epoch))
		for batch_idx, (imgs, target) in enumerate(dataloader):
			
			valid = torch.ones(imgs.size(0), 1)
			valid = valid.to(device)
			fake = torch.zeros(imgs.size(0), 1).to(device)
			fake = fake.to(device)
			imgs = imgs.to(device)
			
			#train generator
			optimizer_G.zero_grad()
			z = torch.randn(imgs.size(0), args.latent_dim)
			z = z.to(device)
			gen_imgs = generator(z)
			loss_g = adversarial_loss(discriminator(gen_imgs), valid)
			loss_g.backward()
			optimizer_G.step()
			
			#train discriminator
			optimizer_D.zero_grad()
			real_loss = adversarial_loss(discriminator(imgs), valid)
			
			"""
				detach可以关闭一个tensor，使其不被tracking
				
				这里不可以使用gen_imgs.requires_grad=False来指定
				
				报错信息：
					RuntimeError: you can only change requires_grad flags of leaf variables. 
					If you want to use a computed variable in a subgraph that doesn't require 
					differentiation use var_no_grad = var.detach()
					
				官方文档中对于leaf variable的解释是：
					 If the Variable has been created by the user, its creator 
						will beNone and we call such objects leaf Variables
				
				也就是说只有自己创建的tensor才能手动指定requires_grad
				
			"""
			print(gen_imgs.requires_grad)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
			loss_d = (real_loss + fake_loss) / 2
			loss_d.backward()
			optimizer_D.step()
			batches_done = epoch * len(dataloader) + batch_idx
			if batches_done % args.log_interval == 0:
				print("Generator loss:{}, Discriminator loss:{}".format(loss_g.item(), loss_d.item()))
				save_image(gen_imgs.data[:25], 'mnist/generateddata/%d.png' % batches_done, nrow=5, normalize=True)
				
if __name__=="__main__":
	main()