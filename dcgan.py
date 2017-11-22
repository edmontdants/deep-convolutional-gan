import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

epochs = 1
lr = 0.0001
batch_size = 128
# Add your own dataset as data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.disc = nn.Sequential(
			nn.Conv2d(1, 128, 4, 2, 1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 256, 4, 2, 1),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(256),

			nn.Conv2d(256, 512, 4, 2, 1),
			nn.LeakyReLU(0.2),
			nn.BatchNorm2d(512),

			nn.Conv2d(512, 1024, 4, 2, 1),
			nn.BatchNorm2d(1024),

			nn.Conv2d(1024, 1, 4, 1, 0),

			nn.Sigmoid())

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	def forward(self, x):
		output = self.disc(x)
		return output


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.gen = nn.Sequential(
			nn.ConvTranspose2d(100, 1024, 4, 1, 0),
			nn.BatchNorm2d(1024),
			nn.ReLU(),

			nn.ConvTranspose2d(1024, 512, 4, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.ConvTranspose2d(128, 1, 4, 2, 1),
			nn.Tanh())

    # weight_init
	def weight_init(self, mean, std):
	    for m in self._modules:
	        normal_init(self._modules[m], mean, std)

	def forward(self, x):
		output = self.gen(x)
		return output

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

G = Generator()
D = Discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

criterion = nn.BCELoss()

G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

for i in range(epochs):
	for x, _ in data_loader:

		D.zero_grad()
		bz = x.size()[0]
		y_real = Variable(torch.ones(bz))
		y_fake = Variable(torch.zeros(bz))
		x = Variable(x)

		D_result = D(x).squeeze()
		D_real_loss = criterion(D_result, y_real)

		z = Variable(torch.randn((bz, 100)).view(-1, 100, 1, 1))

		G_result = G(z)

		D_result = D(G_result).squeeze()
		D_fake_loss = criterion(D_result, y_fake)
		D_fake_score = D_result.data.mean()

		D_train_loss = D_real_loss + D_fake_loss

		D_train_loss.backward()
		D_optimizer.step()


		G.zero_grad()

		z = Variable(torch.randn((bz, 100)).view(-1, 100, 1, 1))

		G_result = G(z)
		D_result = D(G_result).squeeze()
		G_train_loss = criterion(D_result, y_real)
		G_train_loss.backward()
		G_optimizer.step()
		print(i)

noise = Variable(torch.randn((bz, 100)).view(-1, 100, 1, 1))
sample = G(noise)


