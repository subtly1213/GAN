import numpy as np
import torch
import torchvision
from ignite.metrics.gan import FID
import torchvision.transforms as transforms
import os

from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = transforms.Resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.array(images_list)

transform=torchvision.transforms.Compose([
    transforms.Resize([96,96]),
    transforms.ToTensor(),
])

data = torchvision.datasets.ImageFolder('D:\pythonProject9\GAN', transform)
result = torchvision.datasets.ImageFolder('D:\pythonProject9\\test', transform)
train_dataloader = DataLoader(data, batch_size=10, drop_last=True, shuffle=True)
test_dataloader = DataLoader(result, batch_size=10, drop_last=True, shuffle=True)
print(data[0][0].size)
for data1 in train_dataloader:
	img1, _ =data1
for data2 in test_dataloader:
	img2, _ =data2
print(img1.size)
print(img2.size)
images1 = img1
images2 = img2

device = torch.device("cuda:0")
y_pred, y = images1, images2
y_pred = Variable(y_pred).to(device)
y = Variable(y).to(device)
m = FID()
m.update((y_pred, y))
print(m.compute())