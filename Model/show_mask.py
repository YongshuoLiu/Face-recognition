from PIL import Image
from pylab import *
import torch
import numpy
from torchvision.io import image
from torchvision import utils as vutils



a = torch.ones(1, 720, 1280)
b = torch.zeros(1, 720, 1280)
im = array(Image.open('output.png').convert('L'))  # 打开图像，并转成灰度图像
im = torch.tensor(im)
c = torch.where(im!=0,a,b)
input_tensor = c.clone().detach()
input_tensor = input_tensor.to(torch.device('cpu'))
vutils.save_image(input_tensor, "mask.jpg")
print(c)
